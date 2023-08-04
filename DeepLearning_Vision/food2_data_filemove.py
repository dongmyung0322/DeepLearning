import glob
import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

# org data path
label_folder_path = './data/7_19_ex2'

# new data path
dataset_folder_path = './data/food2_data'

# train, val folder path
train_folder_path = os.path.join(dataset_folder_path, 'train')
val_folder_path = os.path.join(dataset_folder_path, 'val')

# train, val folder create
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

# create label dict
label_dict = {}
txt_file = open('./data/7_19_ex2/class_list.txt')
lines = txt_file.readlines()
for line in lines:
    label_idx = line.split(' ')[0]
    label = line.split(' ')[1]
    label = label[:len(label)-1]
    # print(label_idx, label)
    if label not in label_dict:
        label_dict[label_idx] = label
# print(label_dict)

# create csv dict
train_csv_data = pd.read_csv('./data/7_19_ex2/train_labels.csv')
val_csv_data = pd.read_csv('./data/7_19_ex2/val_labels.csv')
csv_dict = {}
csv_img_list = train_csv_data['img_name'].to_list() + val_csv_data['img_name'].to_list()
csv_label_list = train_csv_data['label'].to_list() + val_csv_data['label'].to_list()
for i in range(len(csv_img_list)):
    csv_dict[csv_img_list[i]] = csv_label_list[i]

# create folder
img_paths = glob.glob(os.path.join(label_folder_path, '*', '*', '*.jpg'))
#print(train_img_paths) ./data/7_19_ex2\\train_set\\train_set\\train_120214.jpg
for img_path in tqdm(img_paths):
    img_name = img_path.split('\\')[-1]
    train_val_folder = img_path.split('\\')[1]
    #train_120215.jpg, train_set
    img_label = csv_dict[img_name]
    label_name = label_dict[str(img_label)]

    train_label_folder_path = os.path.join(train_folder_path, label_name)
    val_label_folder_path = os.path.join(val_folder_path, label_name)

    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)

    if train_val_folder == 'train_set':
        target_path = os.path.join(train_label_folder_path, img_name)  # label img path
        shutil.copyfile(img_path, target_path)
    else:
        target_path = os.path.join(val_label_folder_path, img_name)  # label img path
        shutil.copyfile(img_path, target_path)
