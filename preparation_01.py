#####----------------------------------------------------------------------------------#####
# hieunguyen@genesolutions.vn
# This PREPARATION module is used to define the model architecture and its corresponding tokenizer.
# We also provide the function to read the large input text file into batches.
# The main purpose is to prepare the model for the >>> PRETRAINING <<< process .
# NOT FOR FINE-TUNING YET!!!!
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
from typing import Iterator, List
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

#####-------------------------------------------------------------------#####
# CHECK GPU DEVICE TYPE
# Hieu's note: I'm running this script firstly in my MAC M2, so I will check if the MPS device is available.
# If it is, I will use it to run the model.
# >>>>> Update soon: Test this with the GPU RTX3070.
#####-------------------------------------------------------------------#####
def check_mps_device():
    """
    Checks if the MPS device is available and prints the value of x.

    Returns:
        None
    """
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
        
#####-------------------------------------------------------------------#####
# DEFINE THE BERT MODELS
#####-------------------------------------------------------------------#####
##### input args
# path_to_model_files = "./model_files"
# vocab_dir = "./vocabs"
# model_arc = "bert-large-uncased"
# vocab_version = "DNA_no_methyl"

def define_BERT_model(model_arc="bert-large-uncased", 
                      vocab_version="DNA_no_methyl", 
                      path_to_model_files="./model_files", 
                      vocab_dir="./vocabs"):
    """
    Define a BERT model for DNA sequences. This could include the DNA sequence with methylation information (A T G C mC) or not (T C G A only).

    Args:
        model_arc (str): The architecture of the BERT model. Default is "bert-large-uncased".
        vocab_version (str): The version of the vocabulary file. Default is "DNA_no_methyl".
        path_to_model_files (str): The path to the directory containing model configuration files. Default is "./model_files".
        vocab_dir (str): The directory to store vocabulary files. Default is "./vocabs".

    Returns:
        tuple: A tuple containing the configuration, model class, and tokenizer for the BERT model.
    """
    # generate vocab file (if not exists)
    if os.path.isfile(os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version))) == False:
        # UPDATE ME: Generate more types of vocabs for different lengths, DNA with methyl, ....
        if vocab_version == "DNA_no_methyl":
            letters = ["A", "T", "G", "C", "N"]
            vocabs = ["{}{}{}".format(i, j, k) for i in letters for j in letters for k in letters]
            vocabs = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + vocabs
            pd.DataFrame(data=vocabs).to_csv(os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version)),
                                              sep="\t",
                                              index=False,
                                              header=False)
        else:
            raise FileNotFoundError("Vocab file for version {} does not exist at {}.".format(vocab_version, os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version))))

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

    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]  # model init
    config = config_class.from_pretrained(MODELS[model_arc])

    tokenizer = tokenizer_class(vocab_file=os.path.join(vocab_dir, "{}.vocab.txt".format(vocab_version)),
                                do_lower_case=False,
                                do_basic_tokenize=False,
                                tokenize_chinese_chars=False)
    return config, model_class, tokenizer

#####-------------------------------------------------------------------#####
# DEFINE THE FUNCTION TO READ LARGE INPUT TEXT FILE INTO BATCHES
#####-------------------------------------------------------------------#####
class LineByLineBatchedTextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer, batch_size: int):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        # Efficiently count lines
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for i, _ in enumerate(file, 1):
                pass
        return i

    def __getitem__(self, idx):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == idx:
                    encoded_line = self.tokenizer.encode_plus(
                        line.strip(),
                        add_special_tokens=True,
                        return_tensors='pt',
                        max_length=512,  # Assuming max_length for BERT
                        truncation=True,
                        padding=False
                    )
                    return encoded_line['input_ids'].squeeze(0), encoded_line['attention_mask'].squeeze(0)
        # Return None if idx is out of bounds
        return None

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids_padded, attention_masks_padded


