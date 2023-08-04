import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from art_data_custom_dataset import CustomDatset
from tqdm import tqdm
import cv2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # pt load
    model.load_state_dict(torch.load(f='./best_model/art_best.pt'))

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDatset('./data/art_data/val', val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.to(device)
    model.eval()

    correct = 0
    label_dict = {0: 'Abstract', 1: 'Cubist', 2: 'Expressionist', 3: 'Impressionist', 4: 'Landscape', 5: 'Pop Art',
                  6: 'Portrait', 7: 'Realist', 8: 'Still Life', 9: 'Surrealist'}
    with torch.no_grad():
        for img, label, path in test_loader:
            label_ = label.item()
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()


            # 모델의 사진별 예측값
            image = cv2.imread(path[0])
            image = cv2.resize(image, (500,500))
            # 예측값 이미지 안에 입력
            target_label = label_dict[label_]
            true_label_text = f'True: {target_label}'
            pred_label = label_dict[pred.item()]
            pred_text = f'pred: {pred_label}'
            image = cv2. rectangle(image, (0,0), (500, 80), (255,255,255), -1)
            image = cv2.putText(image, pred_text, (0, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            image = cv2.putText(image, true_label_text, (0, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.imshow('test',image)
            if cv2.waitKey() == ord('q'):
                exit()
            print(image)
            print(pred.item(), path)

    print('Test set: Acc {}/{} [{:.0f}]%\n'.format(correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

if __name__ == '__main__':
    main()
