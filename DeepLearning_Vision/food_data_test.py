import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import  resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from food_data_custom_dataset import CustomDataset
from tqdm import tqdm
import cv2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = resnet50()
    # model.fc = nn.Linear(2048, 20)
    # model.load_state_dict(torch.load(f='./best_model/ex01_0717_resnet50_best.pt'))
    model = mobilenet_v2(pretrained=True)
    in_features = 1280
    model.classifier[1] = nn.Linear(in_features, 20)
    model.load_state_dict(torch.load(f='./best_model/food_data.pt'))

    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    test_dataset = CustomDataset('./data/food_data/test/', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img, label = img.to(device).float(), label.to(device)
            output = model(img)
            _,pred = torch.max(output, 1)
            correct += (pred==label).sum().item()
    print('Test set: Acc {}/{} [{:.0f}]%\n'.format(correct, len(test_loader.dataset),
                                                   100 * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()