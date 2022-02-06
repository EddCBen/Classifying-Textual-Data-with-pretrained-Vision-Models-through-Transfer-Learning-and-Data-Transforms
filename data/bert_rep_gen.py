"""
Script to generate representations for IMDB dataset using the last six layers of 
pre-trained BERT-base model from HuggingFace
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional
from transformers import BertTokenizer, BertModel
from pathlib import Path

imdb_path = Path("./imdb_dataset")
#Data Loading
df = pd.read_csv(imdb_path / 'IMDB.csv')
df = df[['review','sentiment']]
sentences = df['review']

#Tokenizerbert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertModel = BertModel.from_pretrained('bert-base-uncased',
                        output_hidden_states=True)
bertModel.eval()

#Tokenizing and Obtaining 6  Layers-[CLS] and Saving into a file
data_list = []
counter = 0
for sent in sentences:
    print("Embedding Sentences number : {}".format(counter))
    #Obtaining Token ID's
    cls_12layers = []
    encoded_sent = tokenizer.encode_plus(
                                    sent,
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'longest',
                                    truncation = True,
                                    return_attention_mask = True,
                                    return_tensors = 'pt',
                                    return_length = True
                                    )
    with torch.no_grad():
        bertModel.eval()
        output = bertModel.cuda()(encoded_sent['input_ids'].to(torch.device("cuda")))
    hidden_states = output.hidden_states[6:]
    for i,_ in enumerate(hidden_states):
        cls_12layers.append(hidden_states[i].squeeze()[0].cpu()) #6x768
    cls_12layers = torch.stack(cls_12layers)
    data_list.append(cls_12layers)
    counter += 1

torch.save(torch.stack(data_list),"IMDB_cls_last6layers.pt")
