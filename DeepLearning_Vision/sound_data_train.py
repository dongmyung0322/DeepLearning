from torchvision.transforms import transforms
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.models import VGG11_Weights
from torchvision.models import vgg11
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sound_data_custom_dataset import CustomDatset
from tqdm import tqdm

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
            torch.save(model.state_dict(), 'sound_best.pt')
            best_val_acc = val_acc

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Val_Loss: {val_loss:.4f},'
            f'Val_acc: {val_acc:.4f}')

    return model, train_losses, val_losses, train_accs, val_accs
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = vgg11(weights=VGG11_Weights.DEFAULT)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 3)
    model.to(device)

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # dataset
    train_dataset = CustomDatset('./data/sound_data/train', transform=train_transform)
    val_dataset = CustomDatset('./data/sound_data/validation', transform=val_transform)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=100, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, num_workers=4, pin_memory=True)

    # import time
    # import math
    # test = time.time()
    # math.factorial(10000)
    # for data, t in val_loader:
    #     print(data, t)
    # test01 = time.time()
    # print(f'{test01 - test :.5f} sec')

    epochs = 10
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay = 1e-2)

    train(model, train_loader, val_loader, epochs, device, criterion, optimizer)

if __name__ == '__main__':
    main()




