import os
import random
import shutil

# org data path
label_folder_path = './data/7-18_ex2'

# new data path
dataset_folder_path = './data/pneumonia_data'

# train, val folder path
train_folder_path = os.path.join(dataset_folder_path, 'train')
val_folder_path = os.path.join(dataset_folder_path, 'val')

# train, val folder create
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

org_folders = os.listdir(label_folder_path)

for org_folder in org_folders:
    org_folder_full_path = os.path.join(label_folder_path, org_folder)
    imgs = os.listdir(org_folder_full_path)
    random.shuffle(imgs)

    # label folder create
    train_label_folder_path = os.path.join(train_folder_path, org_folder)
    val_label_folder_path = os.path.join(val_folder_path, org_folder)

    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)

    # img -> train folder move
    split_index = int(len(imgs) * 0.9)
    for img in imgs[:split_index]:
        src_path = os.path.join(org_folder_full_path, img) # org img path
        dst_path = os.path.join(train_label_folder_path, img) # label img path
        shutil.copyfile(src_path, dst_path)

    for image in imgs[split_index]:
        src_path = os.path.join(org_folder_full_path, img)  # org img path
        dst_path = os.path.join(val_label_folder_path, img)  # label img path
        shutil.copyfile(src_path, dst_path)
print('done')