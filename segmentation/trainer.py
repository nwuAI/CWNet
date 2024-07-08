import os
import time
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance
from utils.metrics import AverageMeter, get_metrics
# import ttach as tta


class Trainer:
    def __init__(self, model, CFG=None, loss=None, train_loader=None, val_loader=None):
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)   # 为混合精度训练提供了梯度缩放和还原机制，方便进行混合精度训练
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())  # 并行计算
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir = os.path.join(
            CFG.save_dir, self.CFG['model']['type'], start_time)  # 保存checkpoint
        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)  # 将路径数据写入TensorBoard中
        dir_exists(self.checkpoint_dir)  # 判断路径是否存在
        cudnn.benchmark = True  # 根据输入数据大小和形状选择合适的卷积算法，优化运行速度，可能会增加额外开销

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):  # 开始训练
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:  # val_loader存在并且 epoch除以val_per_epochs余数为0
                results = self._valid_epoch(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            if epoch % self.CFG.save_period == 0:  # 一个训练周期进行保存一次
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        wrt_mode = 'train'
        self._reset_metrics()   # 指标重置
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):
                    pre = self.model(img)
                    loss = self.loss(pre, gt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre = self.model(img)
                loss = self.loss(pre, gt)
                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        predict = self.model(img)
                        loss = self.loss(predict, gt)
                else:
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()
    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }
