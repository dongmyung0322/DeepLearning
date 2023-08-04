from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from food_data_custom_dataset import CustomDataset
from tqdm import tqdm
import pandas as pd

def train(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    print('Train........')

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        model.train()

        train_loader_iter = tqdm(train_loader, desc=(f'Epoch: {epoch+1}/{epochs}'), leave=False)
        for i, (img,label) in enumerate(train_loader_iter):
            img, label = img.float().to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(output, 1)
            train_acc += (pred == label).sum().item()

            train_loader_iter.set_postfix({'Loss': loss.item()})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for img,label in val_loader:
                img, label = img.float().to(device), label.to(device)
                output = model(img)
                _, pred = torch.max(output, 1)
                val_acc += (pred == label).sum().item()
                val_loss += criterion(output, label).item()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader.dataset)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # save model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), './best_model/food_data.pt')
            best_val_acc = val_acc

    print(f"Epoch [{epoch + 1} / {epochs}] , Train loss [{train_loss:.4f}],"
          f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
          f"Val ACC [{val_acc:.4f}]")

    # visualize
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('food_loss_plot.png')

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('food_acc_plot.png')

    df = pd.DataFrame({
        'Train Loss': train_losses,
        'Train Accuracy': train_accs,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accs
    })
    df.to_csv('food_train_val_result.csv', index=False)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilenet_v2(pretrained=True)
    in_features = 1280
    model.classifier[1] = nn.Linear(in_features, 20)

    model.to(device)

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6),
        A.RandomShadow(),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    train_dataset = CustomDataset('./data/food_data/train', transform=train_transform)
    val_dataset = CustomDataset('./data/food_data/validation/', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=124, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=124)

    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    train(model, train_loader, val_loader, epochs, criterion, optimizer, device)

if __name__ == '__main__':
    main()