import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

image_folder_path = './data/7_24_ex1/images/'
annotation_folder_path = './data/7_24_ex1/annotations/'

# new folder
train_folder = './data/candy_data/train'
eval_folder = './data/candy_data/eval'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)

csv_file_path = os.path.join(annotation_folder_path, 'annotations.csv')

annotation_df = pd.read_csv(csv_file_path)

# 데이터셋을 이미지 기준으로 분할시키고자 함
image_names = annotation_df['filename'].unique()
train_names, eval_names = train_test_split(image_names, test_size=0.2)

train_annotations = pd.DataFrame(columns=annotation_df.columns)
for image_name in train_names:
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(train_folder, image_name)
    shutil.copy(img_path, new_image_path)

    # annotation csv 생성
    annotation = annotation_df.loc[annotation_df['filename'] == image_name].copy()
    annotation['filename'] = image_name
    train_annotations = train_annotations._append(annotation)

print(train_annotations)
train_annotations.to_csv(os.path.join(train_folder, 'annotations.csv'), index=False)

eval_annotations = pd.DataFrame(columns=annotation_df.columns)
for image_name in eval_names:
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(eval_folder, image_name)
    shutil.copy(img_path, new_image_path)

    # annotation csv 생성
    annotation = annotation_df.loc[annotation_df['filename'] == image_name].copy()
    annotation['filename'] = image_name
    eval_annotations = eval_annotations._append(annotation)

eval_annotations.to_csv(os.path.join(eval_folder, 'annotations.csv'), index=False)