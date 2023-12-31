from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.efficientnet import efficientnet_v2_s
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from mPox_data_custom_dataset import CustomDataset
from lion_pytorch import Lion
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    print('Train........')
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss= 0.0
        val_acc = 0.0

        model.train()

        #tqdm
        train_loader_iter = tqdm(train_loader, desc=(f'Epoch: {epoch+1}/{epochs}'), leave=False)

        for i, (img,label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # acc
            _, pred = torch.max(output, 1)
            train_acc += (pred == label).sum().item()

            train_loader_iter.set_postfix({'Loss': loss.item()})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)

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
            torch.save(model.state_dict(), './best_model/mPox-2')
            best_val_acc = val_acc
        print(f'Epoch [{epoch+1} / {epochs}], Train loss [{train_loss:.4f}], Train acc [{train_acc:.4f}],'
              f'Val loss [{val_loss:.4f}], Val acc[{val_acc:.4f}]')

    return model, train_losses, val_losses, train_accs, val_accs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilenet_v2(pretrained=True)
    in_features = 1280
    model.classifier[1] = nn.Linear(in_features, 15)
    # model = efficientnet_v2_s(pretrained=True)
    # in_features = 1280
    # model.classifier[1] = nn.Linear(in_features, 6)
    model.to(device)

    # aug
    train_transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))
    ])
    val_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    train_dataset = CustomDataset('./data/mPox_data/train/', transform=train_transformer)
    val_dataset = CustomDataset('./data/mPox_data/val/', transform=val_transformer)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True)

    epochs = 50
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Lion(model.parameters(), lr=0.001, weight_decay=1e-2)

    train(model, train_loader, val_loader, epochs, optimizer, criterion, device)


if __name__ == '__main__':
    main()