from torch.utils.data import Dataset
import os
import glob
from PIL import Image

class CustomDatset(Dataset):
    def __init__(self, data_dir, transform=None):
        # data_dir = "./data/sound_data/train"
        self.data_dir = glob.glob(os.path.join(data_dir, '*', '*.png'))
        self.transform = transform
        self.label_dict = {'Abstract':0, 'Cubist':1, 'Expressionist':2, 'Impressionist':3, 'Landscape':4, 'Pop Art':5,
                           'Portrait':6, 'Realist':7, 'Still Life':8, 'Surrealist':9}

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, item):
        img_path = self.data_dir[item]
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES =True
        img = Image.open(img_path)
        img = img.convert('RGB')
        label_name =img_path.split('\\')[1]
        label = self.label_dict[label_name]


        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path

#test = CustomDatset('./data/art_data/val', transform=None)

#for img, label in test:
    #print(img,label)