#####--------------------------------------------------------------#####
# T  E S T - F U N C T I O N - M A S K - T O K E N S
#####--------------------------------------------------------------#####

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
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from preparation import *
torch.cuda.empty_cache()

n_process = 1

MASK_LIST = [-1, 1, 2]


#####--------------------------------------------------------------#####
##### I N P U T - A R G U M E N T S
#####--------------------------------------------------------------#####

##### PATHS
outdir = "/mnt/WORKDIR/hieunguyen/outdir/TCR_antigen_binding_transformers/pytorch"
input_datadir = os.path.join(outdir, "workdir")
path_to_input_dataset = os.path.join(input_datadir, "pretraining_data.txt")

output_datadir = os.path.join(outdir, "output")
os.system("mkdir -p {}".format(output_datadir))

global_batch_training_output = os.path.join(output_datadir, "global_batch_training_output")
os.system("mkdir -p {}".format(global_batch_training_output))
##### PARAMETERS
resume_checkpoint_dir = None

per_gpu_train_batch_size = 16
per_gpu_eval_batch_size = 16
max_steps = 10000
gradient_accumulation_steps = 10
save_steps = 100
evaluation_step = 100
max_grad_norm = 1

# local_rank = -1 --> no distributed learning, a single machine can have multiple GPUs
# local_rank != -1 ---> distributead learning, cluster/node structure, each node has a single GPU
local_rank = -1 

weight_decay = 0
learning_rate = 5e-5
adam_epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999
warmup_steps = 0

# Load the model from checkpoint if checkpoit exists
input_dataset = load_and_cache_examples(path_to_input_dataset, tokenizer, n_process = n_process)

train_size = int(0.8 * len(input_dataset.examples))
validation_size = len(input_dataset.examples) - train_size

train_dataset, eval_dataset = torch.utils.data.random_split(input_dataset, [train_size, validation_size])

model, global_step, tr_loss = train(  train_dataset = train_dataset,
                                      eval_dataset = eval_dataset,
                                      evaluation_step = evaluation_step,
                                      per_gpu_train_batch_size = per_gpu_train_batch_size,
                                      per_gpu_eval_batch_size = per_gpu_eval_batch_size,
                                      max_steps = max_steps,
                                      save_steps = save_steps,
                                      gradient_accumulation_steps = gradient_accumulation_steps,
                                      max_grad_norm = max_grad_norm,
                                      local_rank = local_rank, 
                                      weight_decay = weight_decay, 
                                      learning_rate = learning_rate,
                                      adam_epsilon = adam_epsilon,
                                      beta1 = beta1, 
                                      beta2 = beta2,
                                      warmup_steps = warmup_steps,
                                      output_dir = output_datadir,
                                      resume_checkpoint_dir = resume_checkpoint_dir)

           

model_to_save = ( model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
model_to_save.save_pretrained(global_batch_training_output)
tokenizer.save_pretrained(global_batch_training_output)
