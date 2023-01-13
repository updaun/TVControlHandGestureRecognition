import numpy as np
import cv2

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from config import CFG

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            _, img = cap.read()
            img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)

def split_train_val(df, val_ratio):
    train, val, _, _ = train_test_split(df, df['label'], test_size=val_ratio, random_state=CFG['SEED'])

    return train, val

def load_data(data, shuffle=False, test_mode=False):
    if test_mode:
        dataset = CustomDataset(data['path'].values, None)
    else:
        dataset = CustomDataset(data['path'].values, data['label'].values)
    
    dataloader = DataLoader(dataset, batch_size = CFG['BATCH_SIZE'], shuffle=shuffle, num_workers=0)

    return dataloader

def split_load_train_val(df, val_ratio):
    train, val = split_train_val(df, val_ratio)

    train_loader = load_data(train, shuffle=True)
    val_loader = load_data(val)

    return train_loader, val_loader