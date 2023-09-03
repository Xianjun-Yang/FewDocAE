import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import LEDTokenizer, LEDModel
from transformers import BertTokenizer, BertModel

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path): 
        nn.Module.__init__(self)
        if pretrain_path[-5:] == '16384':
            self.bert = LEDModel.from_pretrained(pretrain_path)
        else:
            self.bert = BertModel.from_pretrained(pretrain_path)

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True)
        word_embeddings = outputs.last_hidden_state
        del outputs
        return word_embeddings
