#####----------------------------------------------------------------------------------#####
# MODEL DEFINITION
# hieunguyen@genesolutions.vn
#####----------------------------------------------------------------------------------#####

'''
Run this function to define the model architecture and its corresponding tokenizer.
'''
from optimization import *
from configuration_bert import *
from tokenization_bert import *
from modeling_bert import *

import pathlib
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
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F
from time import sleep

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

##### input args
# path_to_model_files = "./model_files"
# vocab_dir = "./vocabs"
# model_arc = "bert-large-uncased"
# vocab_version = "DNA_no_methyl"

def define_BERT_model(model_arc = "bert-large-uncased", 
                      vocab_version = "DNA_no_methyl", 
                      path_to_model_files = "./model_files", 
                      vocab_dir = "./vocabs"):
    
    #####  generate vocab file (if not exists)
    if os.path.isfile(os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version))) == False:
        # UPDATE ME: Generate more tyeps of vocabs for different lengths, DNA with methyl, ....
        if vocab_version == "DNA_no_methyl":
            letters = ["A", "T", "G", "C", "N"]
            vocabs = ["{}{}{}".format(i, j, k) for i in letters for j in letters for k in letters]
            vocabs = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + vocabs
            pd.DataFrame(data = vocabs).to_csv(os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version)),
                                              sep = "\t",
                                              index = False,
                                              header = False)
        else:
            print("Vocab file for version {} existed at {}.".format(vocab_version, os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version))))
    
    # Path to model configuration, stored in json files
    all_models = [item for item in pathlib.Path(path_to_model_files).glob("*/config.json")]
    MODELS = dict()
    for file in all_models:
        model_name = str(file).split("/")[-2]
        MODELS[model_name] = str(file)
    
    # Define MODEL CLASSES, here we use the basic BERT MLM only, no NSP,
    # UPDATE ME!!!! CHECK THE PERFORMANCE IF USING NSP + MLM?
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForMaskedLM, BertTokenizer)
    }
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"] # model init
    config = config_class.from_pretrained(MODELS[model_arc])
    
    tokenizer = tokenizer_class(vocab_file = os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version)), 
                                do_lower_case = False, 
                                do_basic_tokenize = False, 
                                tokenize_chinese_chars = False)
    return config, model_class, tokenizer
#####----------------------------------------------------------------------------------#####
# Loading dataset line by line, text dataset, into object with tokenization
#####----------------------------------------------------------------------------------#####
class LineByLineTextDataset(Dataset):        
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, n_process = 1, block_size = 512):
        assert os.path.isfile(file_path)
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

##### collate function
def collate(examples: List[torch.Tensor]):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    output = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    return output
