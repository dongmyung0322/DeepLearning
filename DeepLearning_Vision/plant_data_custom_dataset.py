from torch.utils.data import Dataset
import os
import glob
from PIL import Image, ImageFile

class CustomDataset(Dataset):
    def __init__(self, dir_path, transform):
        self.dir_path = glob.glob(os.path.join(dir_path, '*', '*.jpg'))
        # dir_path = './data/plant_data/train/'
        self.transform = transform
        self.label_dict = {'Carpetweeds': 0, 'Crabgrass': 1, 'Eclipta': 2, 'Goosegrass': 3, 'Morningglory':4 ,
                           'Nutsedge': 5, 'PalmerAmaranth': 6, 'Prickly Sida': 7, 'Purslane': 8, 'Ragweed': 9,
                           'Sicklepod': 10,'SpottedSpurge': 11, 'SpurredAnoda': 12, 'Swinecress' : 13, 'Waterhemp': 14}

    def __getitem__(self, item):
        img_path = self.dir_path[item]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label_name = img_path.split('\\')[1]
        label = self.label_dict[label_name]

        return img, label


    def __len__(self):
        return len(self.dir_path)
    
# test = CustomDataset('./data/plant_data/train', transform=None)
# for a,b in test:
#     print(a,b)