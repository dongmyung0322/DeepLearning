import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import cv2

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # for i, item in enumerate(self.data['filepaths']):
        #     if item.endswith('.lnk'):
        #         self.data.drop(index=i, axis=0, inplace=True)
        #         self.data.reset_index(inplace=True)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx, 1]
        img_path = os.path.join('./data/sport_data/', img_path)
        label = self.data.iloc[idx, 0]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.data)

# csv_file = './data/sport_data/sports.csv'
# dataset = CustomDataset(csv_file, transform=None)
#
# for i in dataset:
#     print(i)
