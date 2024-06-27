from optimization import *
from configuration_bert import *
from tokenization_bert import *
from modeling_bert import *
from tokenization_dna import *
import pathlib
import pandas as pd
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##### HELPER FUNCTIONS
def generate_embedding_matrix(path_to_input_dataset, batch_size, model, tokenizer):
    input_dataset = load_and_cache_examples(path_to_input_dataset, tokenizer, n_process = n_process)
    transform_dataloader = DataLoader(input_dataset, batch_size=batch_size, collate_fn=collate)
    
    cls_embeddings = np.empty((0, 1024))
    epoch_iterator = tqdm(transform_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch = batch.to(device)
        model.eval()
        outputs = model(batch)
        last_hidden_states = outputs[0]  
        tmp_cls_embeddings = last_hidden_states[:, 0, :]
        tmp_cls_embeddings = tmp_cls_embeddings.cpu().detach().numpy()
        cls_embeddings = np.vstack([cls_embeddings, tmp_cls_embeddings])
    return cls_embeddings