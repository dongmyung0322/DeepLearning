import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from plant_data_custom_dataset import CustomDataset
from tqdm import tqdm
import cv2


def main():
    label_dict = {0: 'Carpetweeds',1: 'Crabgrass', 2: 'Eclipta', 3: 'Goosegrass', 4: 'Morningglory',
                               5: 'Nutsedge', 6: 'PalmerAmaranth', 7: 'Prickly Sida', 8: 'Purslane', 9: 'Ragweed',
                               10: 'Sicklepod', 11:'SpottedSpurge', 12: 'SpurredAnoda', 13: 'Swinecress', 14: 'Waterhemp'}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilenet_v2()
    in_features = 1280
    model.classifier[1] = nn.Linear(in_features, 15)

    # model load
    model.load_state_dict(torch.load(f='./best_model/plant_best'))
    # print(list(model.parameters()))

    val_transformer = transforms.Compose([
        transforms.CenterCrop((244, 244)),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset('./data/plant_data/val/', transform=val_transformer)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for img, label, img_path in test_loader:
            label_ = label.item()
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()

            target_label = label_dict[label_]
            pred_label = label_dict[pred.item()]
            true_label_text = f'true: {target_label}'
            pred_label_text = f'pred: {pred_label}'

            image = cv2.imread(img_path[0])
            image = cv2.resize(image, (500,500))
            image = cv2.rectangle(image, (0, 0), (500, 80), (255, 255, 255), -1)  # -1 >> 색으로 채워 넣기
            image = cv2.putText(image, pred_label_text, (0, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            image = cv2.putText(image, true_label_text, (0, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.imshow('test', image)
            if cv2.waitKey() == ord('q'):
                exit()

    print('Test set: Acc {}/{} [{:.0f}]%\n'.format(correct, len(test_loader.dataset),100 * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()