import torch
import numpy as np
import time
import os
from tqdm import tqdm


class SegLearner:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader= val_loader
        self.args = args

        # resume시 필요한 값
        self.start_epoch = 0
        self.metrics = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'train_miou': [],
            'val_miou': []
        }


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            train_iou = 0.0
            val_iou = 0.0

            for train_i, (img, label) in enumerate(tqdm(self.train_loader)):
                img = img.to(self.device)
                label = label.long().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(img)
                output = output['out']  # deeplab은 output이 dict형태로 쓸 수 있도록 나오므로, 출력치 key로 받아옴
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(output, dim=1)
                train_loss += loss.item()
                correct = torch.sum(pred == label.data)
                # label 이미지와 pred 이미지를 겹쳐 일치하는 픽샐의 갯수가 나옴
                batch_size = img.size(0)
                train_acc += correct.double() / (batch_size * 520 * 520)  # 겹친 픽셀 갯수 / 총 픽셀 갯수

                train_iou += self.calc_iou(pred, label.data)

            _t_loss = train_loss / len(self.train_loader)
            _t_acc = train_acc / len(self.train_loader.dataset)
            _t_iou = train_iou / len(self.train_loader.dataset)
            self.metrics['train_losses'].append(_t_loss)
            self.metrics['train_accs'].append(_t_acc)
            self.metrics['train_miou'].append(_t_iou)

            # eval
            self.model.eval()
            with torch.no_grad():
                for i, (img, label) in enumerate(tqdm(self.val_loader)):
                    img = img.to(self.device)
                    label = label.long().to(self.device)

                    output = self.model(img)
                    output = output['out']
                    loss = self.criterion(output, label)
                    val_loss += loss.item()

                    pred = torch.argmax(output, dim=1)
                    correct = torch.sum(pred == label.data)
                    batch_size = img.size(0)
                    val_acc += correct.double() / (batch_size * 520 * 520)  # 겹친 픽셀 갯수 / 총 픽셀 갯수

                    val_iou += self.calc_iou(pred, label.data)

                _v_loss = val_loss / len(self.val_loader)
                _v_acc = val_acc / len(self.val_loader.dataset)
                _v_iou = val_iou / len(self.val_loader.dataset)
                self.metrics['val_losses'].append(_v_loss)
                self.metrics['val_accs'].append(_v_acc)
                self.metrics['val_miou'].append(_v_iou)

            print(f'Epoch [{epoch + 1} / {self.args.epochs}], Train loss [{_t_loss:.4f}], Train acc [{_t_acc:.4f}],'
                  f' Train iou [{_t_iou:.4f}], Val loss [{_v_loss:.4f}], Val acc[{_v_acc:.4f}],'
                  f'Val iou[{_v_iou:.4f}]')

            self.save_ckpts(epoch)


    def load_ckpts(self,path):  # resume 시 작동, path >> .pt파일 경로
        ckpt_path = os.path.join(self.args.checkpoint_folder_path, self.args.checkpoint_file_name)
        ckpt = torch.load(ckpt_path, path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch']
        self.metrics = ckpt['metrics']

    def calc_iou(self, preds, labels):
        total_iou = 0.0
        for pred, label in zip(preds, labels):
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()

            union_section = np.logical_or(pred, label)
            inter_section = np.logical_and(pred, label)
            union_sum = np.sum(union_section)
            inter_sum = np.sum(inter_section)
            # union_section 과 inter_section은 계산에 사용이 불가능하기에 해당 행렬의 총 픽셀수를 계산하여 사용

            iou = inter_sum / union_sum
            total_iou += iou

        return total_iou

    def save_ckpts(self, epoch):
        if not os.path.exists(self.args.checkpoint_folder_path):
            os.makedirs(self.args.checkpoint_folder_path, exist_ok=True)

        to_save_path = os.path.join(self.args.checkpoint_folder_path, self.args.checkpoint_file_name)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'metrics': self.metrics
        }, to_save_path)
