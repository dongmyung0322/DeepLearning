import glob
import os
import cv2
import random
from tqdm import tqdm

os.makedirs('./data/car_detection_yolo/train/images', exist_ok=True)
os.makedirs('./data/car_detection_yolo/train/labels', exist_ok=True)
os.makedirs('./data/car_detection_yolo/val/images', exist_ok=True)
os.makedirs('./data/car_detection_yolo/val/labels', exist_ok=True)

txt_file_list = glob.glob('./data/car_detection_dataset/train/*.txt')

val_ratio = 0.1
val_size = int(len(txt_file_list) * val_ratio)

val_files = txt_file_list[:val_size]
train_files = txt_file_list[val_size:]

for file in tqdm(train_files):
    file_name = os.path.basename(file)
    # syn_06468.txt
    file_name = file_name.split('.')[0]

    # shutil이 더 효율적
    # create img.png
    img = cv2.imread('./data/car_detection_dataset/train/' + file_name + '.png')
    cv2.imwrite(f'./data/car_detection_yolo/train/images/{file_name}.png', img)

    # create label.txt
    img_height, img_width, _ = img.shape
    with open(file, 'r', encoding='utf-8') as f:
        lines_list = []
        lines = f.readlines()
        for line in lines:
            line = list(map(float,line.strip().split(' ')))
            class_name = int(line[0])
            x_min = float(min(line[5], line[7]))
            y_min = float(min(line[6], line[8]))
            x_max = float(max(line[1], line[3]))
            y_max = float(max(line[2], line[4]))

            x_center = float(((x_min + x_max) / 2) / img_width)
            y_center = float(((y_min + y_max) / 2) / img_height)
            width = abs(x_max - x_min) / img_width
            height = abs(y_max - y_min) / img_height
            lines_list.append([class_name, x_center, y_center, width, height])

    with open('./data/car_detection_yolo/train/' + file_name + '.txt', 'w') as f:
        for line in lines_list:
            f.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' +str(line[4]) + '\n')

for file in tqdm(val_files):
    file_name = os.path.basename(file)
    # syn_06468.txt
    file_name = file_name.split('.')[0]

    # shutil이 더 효율적
    # create img.png
    img = cv2.imread('./data/car_detection_dataset/train/' + file_name + '.png')
    cv2.imwrite(f'./data/car_detection_yolo/val/images/{file_name}.png', img)

    # create label.txt
    img_height, img_width, _ = img.shape
    with open(file, 'r', encoding='utf-8') as f:
        lines_list = []
        lines = f.readlines()
        for line in lines:
            line = list(map(float, line.strip().split(' ')))
            class_name = int(line[0])
            x_min = float(min(line[5], line[7]))
            y_min = float(min(line[6], line[8]))
            x_max = float(max(line[1], line[3]))
            y_max = float(max(line[2], line[4]))

            x_center = float(((x_min + x_max) / 2) / img_width)
            y_center = float(((y_min + y_max) / 2) / img_height)
            width = abs(x_max - x_min) / img_width
            height = abs(y_max - y_min) / img_height
            lines_list.append([class_name, x_center, y_center, width, height])

    with open('./data/car_detection_yolo/val/' + file_name + '.txt', 'w') as f:
        for line in lines_list:
            f.write(
                str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + str(line[4]) + '\n')