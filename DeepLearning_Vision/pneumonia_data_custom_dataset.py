import os
import cv2
import glob
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir,transform=None):
        # ./data/pneumonia_data/train
        self.data_dir = glob.glob(os.path.join(data_dir, '*', '*.jpeg'))
        self.transform = transform
        self.label_dict = self.create_label_dict()

    def create_label_dict(self):
        label_dict = {}
        for file_path in self.data_dir:
            label = os.path.basename(os.path.dirname(file_path))
            if label not in label_dict:
                label_dict[label] = len(label_dict)

        return label_dict

    def __getitem__(self, item):
        img_file_path = self.data_dir[item]
        img = cv2.imread(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = os.path.basename(os.path.dirname(img_file_path))
        label_idx = self.label_dict[label]

        if self.transform is not None:
            img = self.transform(image=img)['image']
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, label_idx

    def __len__(self):
        return len(self.data_dir)

# test = CustomDataset('./data/pneumonia_data/train', transform=None)
# for a,b in test:
#     print(a,b)