import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from mPox_data_custom_dataset import CustomDataset
from tqdm import tqdm
import cv2


def main():
    label_dict = label_dict = {0: 'Chickenpox', 1: 'Cowpox', 2: 'Healthy', 3: 'HFMD', 4: 'Measles', 5: 'Monkeypox'}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilenet_v2()
    in_features = 1280
    model.classifier[1] = nn.Linear(in_features, 6)

    # model load
    model.load_state_dict(torch.load(f='./best_model/ex02_0714_best_mobilenet_v2.pt'))
    # print(list(model.parameters()))

    val_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    test_dataset = CustomDataset('./data/mPox_data/val/', transform=val_transformer)
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

            # image = cv2.imread(img_path[0])
            # image = cv2.resize(image, (500,500))
            # image = cv2.rectangle(image, (0, 0), (500, 80), (255, 255, 255), -1)  # -1 >> 색으로 채워 넣기
            # image = cv2.putText(image, pred_label_text, (0, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            # image = cv2.putText(image, true_label_text, (0, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            # cv2.imshow('test', image)
            # if cv2.waitKey() == ord('q'):
            #     exit()

    print('Test set: Acc {}/{} [{:.0f}]%\n'.format(correct, len(test_loader.dataset),100 * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()