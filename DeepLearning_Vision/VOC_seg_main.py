import torch
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from VOC_seg_custom_dataset import customVOCSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50
from VOC_seg_train import SegLearner


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 가장 최적의 평균/표준편차 값

    train_transforms = A.Compose([
        A.Resize(520, 520),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(520,520),
        A.Normalize(),
        ToTensorV2()
    ])

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='데이터 경로')
    parser.add_argument('--checkpoint_folder_path', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_file_name', type=str, default='VOC_seg_checkpoint.pt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--resume', action='store_true', help='store_true가 있으면 터미널에서 불려야 작동함' )
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()


    train_dataset = customVOCSegmentation(args.data_path, mode='train', transforms=train_transforms)
    val_dataset = customVOCSegmentation(args.data_path, mode='val', transforms=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = deeplabv3_resnet50(pretrained=False)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    epochs = args.epochs

    learner = SegLearner(model, optimizer, criterion, train_loader, val_loader, args)
    if args.resume:
        learner.load_ckpts()

    learner.train()