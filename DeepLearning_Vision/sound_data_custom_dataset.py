from torch.utils.data import Dataset
import os
import glob
from PIL import Image

class CustomDatset(Dataset):
    def __init__(self, data_dir, transform=None):
        # data_dir = "./data/sound_data/train"
        self.data_dir = glob.glob(os.path.join(data_dir, '*', '*.png'))
        self.transform = transform
        self.label_dict = {'MelSepctrogram': 0, 'STFT': 1, 'waveshow': 2}

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, item):
        img_path = self.data_dir[item]
        img = Image.open(img_path)
        img = img.convert('RGB')
        label_name =img_path.split('\\')[1]
        label = self.label_dict[label_name]


        if self.transform is not None:
            img = self.transform(img)

        return img, label

#test = CustomDatset('./data/sound_data/train', transform=None)

#for i in test:
    #print(i)
