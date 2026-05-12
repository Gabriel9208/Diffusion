import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../', '../', 'src'))

import torch
from model.condition_encoder.BaseConditionEncoder import BaseConditionEncoder

class CLIPTextEncoder(BaseConditionEncoder):
    def __init__(self, tokenizer: int, text_encoder: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def forward(self, text_prompt: list):
        text_prompt = [" ".join(caption) for caption in text_prompt]

        text_inputs = self.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
            
        input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
                
        pooled_output = outputs.pooler_output 
            
        return pooled_output