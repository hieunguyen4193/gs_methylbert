from optimization import *
from configuration_bert import *
from tokenization_bert import *
from modeling_bert import *
from tokenization_dna import *

import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F
from time import sleep
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

#####--------------------------------------------------------------#####
##### C A L  L - T H E  - M O  D E L - A N D - T O K E N I Z E R
#####--------------------------------------------------------------#####
MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer)}

VOCAB_FILES = {
    "baseline_vocab": "TCR_antigen_vocab.txt"
}

MODEL_CONFIGS = {
    "bert-large-cased": "bert-large-cased-config.json"
}

outdir = "/mnt/WORKDIR/hieunguyen/outdir/TCR_antigen_binding_transformers/pytorch"
path_to_model_files = os.path.join(outdir, "model_files")

vocab_filename = VOCAB_FILES["baseline_vocab"]
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]

path_to_model_config = os.path.join(path_to_model_files, "models", MODEL_CONFIGS["bert-large-cased"])
config = config_class.from_pretrained(path_to_model_config)

path_to_tokenizer = os.path.join(path_to_model_files, "vocab", vocab_filename)
tokenizer = tokenizer_class(vocab_file = path_to_tokenizer, do_lower_case = False, do_basic_tokenize = False, tokenize_chinese_chars = False)

#####--------------------------------------------------------------#####
# C L A S S E S - D E F I N I T I O N
#####--------------------------------------------------------------#####
##### Load the dataset into batches
class LineByLineTextDataset(Dataset):        
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, n_process = 1, block_size = 512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        
        directory, filename = os.path.split(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        if n_process == 1:
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        else:
            n_proc = n_process
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(lines)/n_proc)
            for i in range(1, n_proc+1):
                if i != n_proc:
                    indexes.append(len_slice*(i))
                else:
                    indexes.append(len(lines))
            results = []
            for i in range(n_proc):
                results.append(p.apply_async(convert_line_to_example,[tokenizer, lines[indexes[i]:indexes[i+1]], block_size,]))
                print(str(i) + " start")
            p.close() 
            p.join()

            self.examples = []
            for result in results:
                ids = result.get()
                self.examples.extend(ids)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

#####--------------------------------------------------------------#####
##### HELPER FUNCTIONS
#####--------------------------------------------------------------#####
def load_and_cache_examples(path_to_input_dataset, tokenizer, block_size = 512, n_process = 1):
    return LineByLineTextDataset(tokenizer, file_path = path_to_input_dataset, block_size = block_size)

def collate(examples: List[torch.Tensor]):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    output = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    return output
    
def preparate_batch_data(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    mask_list = [-2, -1]
    
    labels = inputs.clone()
    
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_index==1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True
    

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


#####--------------------------------------------------------------#####
# T R A I N I N G - F U N C T I O N
#####--------------------------------------------------------------#####
def evaluate_during_training(model,
                            tokenizer, 
                            eval_dataset,
                            device,
                            per_gpu_eval_batch_size,
                            n_gpu):
    logging.info("====================================================")
    logging.info("RUNNING EVALUATION")
    logging.info("====================================================")
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler = eval_sampler, batch_size = eval_batch_size, collate_fn = collate
    )
    
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", eval_batch_size)
                                
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, mlm_labels = preparate_batch_data(batch, tokenizer)
        inputs = inputs.to(device)
        mlm_labels = mlm_labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels = mlm_labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {"perplexity": perplexity,
             "eval_loss": eval_loss}
    
    logger.info("***** Eval results *****")
    
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
    return result
                                
def train(train_dataset,
          eval_dataset,
          evaluation_step,
          per_gpu_train_batch_size,
          per_gpu_eval_batch_size,
          max_steps,
          save_steps,
          gradient_accumulation_steps,
          max_grad_norm,
          local_rank, 
          weight_decay, 
          learning_rate,
          adam_epsilon,
          beta1, 
          beta2,
          warmup_steps,
          output_dir,
          resume_checkpoint_dir):

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    if (resume_checkpoint_dir is None):
        model = model_class(config=config)
    else:
        model = model_class.from_pretrained(
                resume_check_point_dir,
                config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    model = model.to(device)

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset) 
    
    train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate
        )
    
    t_total = max_steps
    num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, betas=(beta1,beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if (resume_checkpoint_dir is not None):
        optimizer.load_state_dict(torch.load(os.path.join(resume_checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(resume_checkpoint_dir, "scheduler.pt")))
             
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if (resume_checkpoint_dir is not None):
        checkpoint_suffix = resume_checkpoint_dir.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

    logging.info("########################################################")
    logging.info("***** Running training *****")
    logging.info("  Using {} GPUs".format(n_gpu))
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
        * gradient_accumulation_steps
        * (torch.distributed.get_world_size() if local_rank != -1 else 1),
    )
    logging.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    logging.info("########################################################")
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()
    
    # Load num_train_epochs to an iterator (tqdm)
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch"
    )
    
    for _ in train_iterator:
        # inside each epoch, load number of steps, based on batch size.
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs, mlm_labels = preparate_batch_data(batch, tokenizer)
            inputs = inputs.to(device)
            mlm_labels = mlm_labels.to(device)
            model.train()
            outputs = model(inputs, masked_lm_labels = mlm_labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            
            # if we have more than 1 GPU --> distributed learning --> average loss in all GPU devices
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward() # compute backward loss.
            tr_loss += loss.item()

            epoch_iterator.set_postfix(loss = loss.item())
            sleep(0.1)
            
            if (step + 1) % gradient_accumulation_steps == 0: # if step reach where we need to do gradient accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # logging.info("Training loss: {}".format(loss))
                ##### EVALUATE DURING TRAINING
                if global_step % evaluation_step == 0:
                    eval_result = evaluate_during_training( model,
                                                        tokenizer, 
                                                        eval_dataset,
                                                        device,
                                                        per_gpu_eval_batch_size,
                                                        n_gpu)
                ##### SAVE CHECKPOINTS
                if global_step % save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    checkpoint_dir = os.path.join(output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.system("mkdir -p {}".format(checkpoint_dir))
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    
                    model_to_save.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logging.info("Saving model checkpoint to %s", checkpoint_dir)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                    logging.info("Saving optimizer and scheduler states to %s", checkpoint_dir)
                    
            if max_steps > 0 and global_step > max_steps:
                    epoch_iterator.close()
                    break
        if max_steps > 0 and global_step > max_steps:
                train_iterator.close()
                break
    return model, global_step, tr_loss / global_step