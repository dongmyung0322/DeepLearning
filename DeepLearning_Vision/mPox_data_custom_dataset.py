from torch.utils.data import Dataset
import os
import glob
from PIL import Image, ImageFile


class CustomDataset(Dataset):
    def __init__(self, dir_path, transform):
        self.dir_path = glob.glob(os.path.join(dir_path, '*', '*.jpg'))
        # dir_path = './data/mPox_data/train/'
        self.transform = transform
        self.label_dict = {'Chickenpox': 0, 'Cowpox': 1, 'Healthy': 2, 'HFMD': 3, 'Measles': 4, 'Monkeypox': 5}

    def __getitem__(self, item):
        img_path = self.dir_path[item]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label_name = img_path.split('\\')[1]
        label = self.label_dict[label_name]

        return img, label, img_path

    def __len__(self):
        return len(self.dir_path)

# test = CustomDataset('./data/mPox_data/train', transform=None)
# for a,b in test:
#     print(a,b)