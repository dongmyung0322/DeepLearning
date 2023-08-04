import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg11
from sound_data_custom_dataset import CustomDatset
from tqdm import tqdm
import cv2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = vgg11()
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 3)

    # pt load
    model.load_state_dict(torch.load(f='./best_model/sound_best.pt'))

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDatset('./data/sound_data/validation', val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # for a,b in test_loader:
    #     print(a,b)
    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()


    print('Test set: Acc {}/{} [{:.0f}]%\n'.format(correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

if __name__ == '__main__':
    main()