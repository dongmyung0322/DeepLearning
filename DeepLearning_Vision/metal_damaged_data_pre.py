import cv2
import os
import glob
import json
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def expend2square(pil_img, background_color):
    width, height = pil_img.size

    if width == height:
        return pil_img

    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize_with_padding(pil_img, newsize, background_color):
    img = expend2square(pil_img, background_color)
    #img = img.resize((newsize[0], newsize[1]), Image.ANTIALIAS)
    img = F.resize(img,(newsize[0], newsize[1]))
    return img

img_path = glob.glob(os.path.join('./data/7_17_ex2/images', '*.jpg'))
json_path = ('./data/7_17_ex2/anno/annotation.json')
with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
print(json_data['img_01_3402617700_00001.jpg'])
count = 0

for img in tqdm(img_path):
    img_name = img.split('\\')[-1]
    img_json = json_data[img_name]
    anno_info = img_json['anno']
    img = Image.open(img).convert('RGB')

    for bbox_idx, bbox_info in enumerate(anno_info):
        label = bbox_info['label']
        bbox = bbox_info['bbox']
        x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        img_cropped = img.crop((x,y,w,h))
        img_resized = resize_with_padding(img_cropped, (255,255), 0)
        # plt.imshow(img_resized)
        # plt.show()

        train_save_path = f'./data/metal_damaged_data/train/{label}'
        val_save_path = f'./data/metal_damaged_data/val/{label}'

        if np.random.rand() < 0.9 :
            os.makedirs(train_save_path, exist_ok=True)
            img_resized.save(f'./data/metal_damaged_data/train/{label}/{img_name}_{bbox_idx}.png')

        else:
            os.makedirs(val_save_path, exist_ok=True)
            img_resized.save(f'./data/metal_damaged_data/val/{label}/{img_name}_{bbox_idx}.png')

        count +=1




