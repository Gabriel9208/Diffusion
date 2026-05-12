import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../', '../', 'src'))

from model.condition_encoder.BaseConditionEncoder import BaseConditionEncoder

import torch
import torch.nn as nn
import json

class MultiHotEncoder(BaseConditionEncoder):
    def __init__(self, mapping_file: str, embed_dim: int):
        super().__init__()
        
        with open(mapping_file, 'r') as f:
            self.label_mapping = json.load(f)
        
        self.vocab_size = len(self.label_mapping)
        
        # experimental
        self.proj = nn.Linear(self.vocab_size, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0)
        #self.mlp = nn.Sequential(
        #    nn.Linear(self.vocab_size, embed_dim),
        #    nn.SiLU(),
        #    nn.Linear(embed_dim, embed_dim)
        #)

    def forward(self, text_prompt: list, pure=False):
        device = next(self.parameters()).device
        
        try:
            labels = [[self.label_mapping[obj] for obj in t] for t in text_prompt]
        except KeyError as e:
            raise ValueError(f"Label {e} not found in mapping file")
         
        indices = torch.tensor([(i, obj) for i, lbl in enumerate(labels) for obj in lbl], device=device)
        
        multihot = torch.zeros(len(labels), self.vocab_size, device=device)
        multihot[indices[:, 0], indices[:, 1]] = 1

        if pure:
            return multihot
        
        emb = self.proj(multihot)
        
        return emb