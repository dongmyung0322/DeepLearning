import torch
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
import os

# input in batch index -> model
def collate_fn(batch):
    imgs, target_boxes, target_labels = tuple(zip(*batch))

    # img list image -> torch.stack use one tensor dim
    imgs = torch.stack(imgs, 0)
    targets = []  # target_boxes와 target_lables를 받기위해

    # target boxes
    for i in range(len(target_boxes)):
        target = {
            'boxes': target_boxes[i],
            'labels': target_labels[i]
        }
        targets.append(target)
    return imgs, targets

class CustomDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.img_path = sorted(glob.glob(os.path.join(root, '*.png')))
        if train:
            self.boxes = sorted(glob.glob(os.path.join(root, '*.txt')))

    def parse_boxes(self, box_path):
        with open(box_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            boxes = []
            labels = []

            for line in lines:
                values = list(map(float, line.strip().split(' ')))
                # [27.0, 394.0, 480.0, 669.0, 480.0, 669.0, 811.0, 394.0, 811.0]
                class_id = int(values[0])
                x_min, y_min = int(round(values[1])), int(round(values[2]))
                x_max = int(round(max(values[3], values[5], values[7])))
                y_max = int(round(max(values[4], values[6], values[8])))

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, item):
        img_path = self.img_path[item]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        if self.train:
            box_path = self.boxes[item]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1  # two station 모델의 경우 0 == background이기 때문에 하나씩 밀기

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed['image']

            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.img_path)


# if __name__ == '__main__':
#     train_dataset = CustomDataset('./data/car_detection_dataset/train', train=True, transforms=None)
#     for i in train_dataset:
#         print(i)