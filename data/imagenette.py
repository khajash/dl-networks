import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

"""
A note on the input size for AlexNet:
As a fun aside, if you read the actual paper it claims that the input images were 224x224, 
which is surely incorrect because (224 - 11)/4 + 1 is quite clearly not an integer. This has 
confused many people in the history of ConvNets and little is known about what happened. My own 
best guess is that Alex used zero-padding of 3 extra pixels that he does not mention in the paper.
- Andrej Karpathy in https://cs231n.github.io/convolutional-networks/ 
"""

IMAGENET_RGB_MEANS = [0.485, 0.456, 0.406]
IMAGENET_RGB_STDS = [0.229, 0.224, 0.225]

LABELS_NUM_MAP = {
    'n02979186': 0,
    'n03417042': 1,
    'n01440764': 2,
    'n02102040': 3,
    'n03028079': 4,
    'n03888257': 5,
    'n03394916': 6,
    'n03000684': 7,
    'n03445777': 8,
    'n03425413': 9
}

class ImagenetteDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        csv_filename: str = "noisy_imagenette.csv",
        train: bool = True,
        noisy_perc: int = 0,
        transform = None,
        target_transform = None,
    ):
        
        csv_data = pd.read_csv(os.path.join(root_dir, csv_filename))
        self.img_paths, self.labels = self.get_dataset(csv_data, train, noisy_perc)
        self.root_dir = root_dir
        
        self.transform = transform
        self.target_transform = target_transform

    def get_dataset(self, csv_data, train, noisy_perc):
        if train:
            data = csv_data[csv_data['is_valid'] == False]
        else:
            data = csv_data[csv_data['is_valid'] == True]
        
        if noisy_perc not in [0,1,5,25,50]:
            raise ValueError(f'{noisy_perc} not a valid noisy label percentage. Select: 0,1,5,25,50')
        
        labels = data[f'noisy_labels_{noisy_perc}']
        labels = labels.replace(LABELS_NUM_MAP)
        img_paths = data['path']
        return img_paths, labels
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # open image as torch tensor
        img_path = os.path.join(self.root_dir, self.img_paths.iloc[idx])
        image = read_image(img_path) # reads image into torch format (C, H, W)

        # make grayscale image 3 channels
        if image.shape[0] == 1:
            image = torch.cat([image]*3, axis=0) 
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels.iloc[idx]
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

