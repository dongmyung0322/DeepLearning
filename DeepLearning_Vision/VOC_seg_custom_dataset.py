from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from PIL import Image

class customVOCSegmentation(VOCSegmentation):
    def __init__(self, root, mode='train', transforms=None):
        self.root = root
        super().__init__(root=self.root, image_set=mode, download=self.check_if_path_exists(), transforms=transforms)
        # torchvision에 이미 존재하는 데이터셋의 경우 custom dataset에서 필요한 인자를 모두 정의하고 있음

    def __getitem__(self, idx):
        # 부모 클래스에 self.images에 이미지가 self.masks에 라벨이 있음
        # 해당 리스트는 이미지 파일 경로만을 담고 있기에 imread로 읽어줘야함
        img = cv2.imread(self.images[idx])
        mask = np.array(Image.open(self.masks[idx]))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask) # cv2사용을 전제로 albumentation 모델 사용
            img = augmented['image']
            mask = augmented['mask']

            mask[mask > 20] = 0

        return img, mask

    def check_if_path_exists(self):
        return False if os.path.exists(self.root) else True
        # self.root에 데이터가 이미 다운되어있으면 False로 다운로드 받을 필요 없다고 보냄

    # __len__의 경우 부모 클래스에서 정의 되었기 때문에 추가 정의 필요 없음

if __name__ == '__main__':
    dataset = customVOCSegmentation('./data')
    for item in dataset:
        img, mask = item
        # print(img, mask)
        # cv2.imshow('org', img)
        # cv2.imshow('mask', mask)
        summary = cv2.copyTo(img, mask)
        marked = cv2.addWeighted(img, 0.5, summary, 0.5, 0)
        cv2.imshow('marked', marked)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            exit()