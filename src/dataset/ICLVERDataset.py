import os
import json
import pandas as pd
import cv2
from torch.utils.data import Dataset


class ICLVERDataset(Dataset):
    def __init__(self, dictionary: dict, img_dir: str, transforms=None):
        super().__init__()

        self.img_dir = img_dir
        self.transforms = transforms

        self.df = pd.DataFrame({
            'path': dictionary.keys(),
            'label': dictionary.values()
        })
        
        self.df['path'] = self.df['path'].apply(lambda x: os.path.join(self.img_dir, x))

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)
            
        label = self.df.iloc[idx]['label']
        
        return img, label 
    
    def __len__(self):
        return len(self.df)