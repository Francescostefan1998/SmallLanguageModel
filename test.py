from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")

# Data preprocessing
# Byte Pour Encoding: it takes a dataset and it gives the list of tokens (subwords)

import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm

enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example['text'])
    out = {'ids': ids, 'len': len(ids)} # we are converting into dictionary of ids and length
    return out

if not os.path.exists("train.bin"):
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
    )