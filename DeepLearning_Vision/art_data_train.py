from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from art_data_custom_dataset import CustomDatset
from tqdm import tqdm
from lion_pytorch import Lion

def train(model, train_loader, val_loader, epochs, device, criterion, optimizer):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    print('Train..........')

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_acc = 0.0

        model.train()

        #tqdm
        train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # acc
            _, pred = torch.max(output, 1)
            train_acc += (pred==label).sum().item()

            # print loss
            if i % 10 == 9:
                #print(f'Epoch [{epoch+1}/{epochs}, Loss: {loss.item()}, Step: [{i+1}/{len(train_loader)}]')
                train_loader_iter.set_postfix({'Loss': loss.item()})

        train_loss /= len(train_loader)
        train_acc = train_acc / len(train_loader.dataset)

        # validate
        model.eval()
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device)

                output = model(img)
                pred = output.argmax(dim=1, keepdim=True)   # >> _, pred = torch.max(output,1)
                val_acc += pred.eq(label.view_as(pred)).sum().item()  # >> (pred==label).sum().item()
                val_loss += criterion(output, label).item()

        val_loss /= len(val_loader)
        val_acc = val_acc / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)


        # save the model with best val acc
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'best.pt')
            best_val_acc = val_acc

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Val_Loss: {val_loss:.4f},'
            f'Val_acc: {val_acc:.4f}')

    return model, train_losses, val_losses, train_accs, val_accs
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.to(device)

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # dataset
    train_dataset = CustomDatset('./data/art_data/train', transform=train_transform)
    val_dataset = CustomDatset('./data/art_data/val', transform=val_transform)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True)

    # import time
    # import math
    # test = time.time()
    # math.factorial(10000)
    # for data, t in val_loader:
    #     print(data, t)
    # test01 = time.time()
    # print(f'{test01 - test :.5f} sec')

    epochs = 20
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = Lion(model.parameters(), lr=0.001, weight_decay = 1e-2)
    # Epoch [20/20], Train_Loss: 0.3863, Train_Acc: 0.8698, Val_Loss: 3.3600,Val_acc: 0.4391
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    # Epoch [20/20], Train_Loss: 0.2284, Train_Acc: 0.9134, Val_Loss: 1.7442,Val_acc: 0.6232

    train(model, train_loader, val_loader, epochs, device, criterion, optimizer)

if __name__ == '__main__':
    main()
