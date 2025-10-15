import argparse
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import load_Seg_Vaihingen
import load_Seg_DDHRNet
import load_WHU
import load_MSRS
import logging
import FocalDiceLoss
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import model_HyFusion
import segmentation_models_pytorch as smp
import load_BraTS
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default = 100
                    , type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency for train (default: 20)')
parser.add_argument('-p-val', '--print-freq-val', default=100, type=int,
                    metavar='N', help='print frequency for val (default: 20)')
parser.add_argument('--resume', default='None',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use.')
parser.add_argument('--input-shape', default=(15, 15),
                    help='size of the input image patch')
parser.add_argument('--aug', default=False,
                    help='use augment or not for inputs')
parser.add_argument('--need-val', default=True,
                    help='use validation or not during training')
# used for adam optimization
parser.add_argument('--use-adam', default=True,
                    help='use adam for training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate of adam', dest='lr')
# used for sgd
parser.add_argument('--lr-sgd', '--learning-rate-sgd', default=0.01, type=float,
                    metavar='LR', help='initial learning rate of sgd', dest='lr')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--numclass', default=8, type=float,
                    help='Number of the classes')

# parser.add_argument('--learner_layers', default=2, type=int,
#                     help='Number of learner layers')
# parser.add_argument('--gcn_layers', default=2, type=int,
#                     help='Number of GCN layers')
# parser.add_argument('--k', default=64, type=int,
#                     help='Value of k')
# parser.add_argument('--nonlinear_idx', default=0, type=int,
#                     help='Nonlinear index')
# parser.add_argument('--dropedge', default=0.2, type=float,
#                     help='Dropedge rate')
# parser.add_argument('--sparse', action='store_true',
#                     help='Enable sparse mode')
# parser.add_argument('--activation', default='relu', type=str,
#                     help='Activation function')
# parser.add_argument('--hidden_dim', default=128, type=int,
#                     help='Hidden dimension')
# parser.add_argument('--emb_dim', default=64, type=int,
#                     help='Embedding dimension')
# parser.add_argument('--proj_dim', default=32, type=int,
#                     help='Projection dimension')
# parser.add_argument('--dropout', default=0.1, type=float,
#                     help='Dropout rate')
# parser.add_argument('--alpha', default=0.5, type=float,
#                     help='Alpha value')




best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    # args.gpu = 'cpu'

    if args.gpu is not None and args.gpu != 'cpu':
        # 确保传入的是整数（如 args.gpu=0）
        device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
        print(f"Use GPU: {args.gpu} for training")
    else:
        device = torch.device("cpu")
        print("Use CPU for training")

    # create model
    # print("=> creating model")

    # model = smp.UnetPlusPlus(encoder_name="resnet34",  # 使用的encoder
    #                   encoder_weights='imagenet',  # 使用预训练权重
    #                   in_channels=3,  # 输入图像的通道数
    #                   classes=7)

    model = model_HyFusion.DualInputHyperbolicUNet(classes=args.numclass)

    print(model)

    model = model.to(device)

    # define the optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    dicefocal = FocalDiceLoss.FocalDiceLoss()

    if args.use_adam:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr_sgd,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= args.epochs, eta_min=1e-5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    # 文件夹路径
    # folderA = '/home/hfz/data/DDHRNet/xian/cloud/GF2'
    # folderB = '/home/hfz/data/DDHRNet/xian/cloud/GF3'
    # folderC = '/home/hfz/data/DDHRNet/xian/cloud/label'

    folderA = '/home/hfz/data/WHU-OPT-SAR/optical256'
    folderB = '/home/hfz/data/WHU-OPT-SAR/sar256'
    folderC = '/home/hfz/data/WHU-OPT-SAR/label256'

    # folderA = '/home/hfz/data/MSRS/vi'
    # folderB = '/home/hfz/data/MSRS/ir'
    # folderC = '/home/hfz/data/MSRS/Segmentation_labels'

    # root_dir = '/home/hfz/data/BraTS2021'

    # 定义变换

    # 创建数据集
    # dataset = load_Seg_DDHRNet.CustomDataset2(folderA, folderB, folderC)
    # dataset = load_MSRS.CustomDataset(folderA, folderB, folderC)
    dataset = load_WHU.CustomDataset4(folderA, folderB, folderC)
    # dataset = load_BraTS.BraTS20213DDataset(root_dir, target_size=(155, 240, 240))

    # 选择数据集的一半
    half_size = len(dataset)
    dataset = random_split(dataset, [half_size, len(dataset) - half_size])[0]

    # 计算训练集和测试集的大小
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # 使用random_split分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger = get_logger('/home/hfz/doc/HyFusion/workdir/exp.log')
    logger.info('start training!')
    for epoch in range(args.start_epoch, args.epochs):
        if not args.use_adam:
            adjust_learning_rate(optimizer, epoch, args)
        # print(optimizer.state_dict()['param_groups'][0]['lr'])

        # train for one epoch
        train(train_loader, model, scheduler, criterion, dicefocal, optimizer, epoch, args)
        if args.need_val:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1 
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best,
                filename= '/home/hfz/doc/HyFusion/workdir/Hyfusion_model_WHU_res_c-1.pth.tar'.format(
                    epoch))
    print(best_acc1)
    logger.info('total_epochs={:.3f}'.format(args.epochs))
    logger.info('batch_size={:.3f}'.format(args.batch_size))
    logger.info('lr={:.3f}'.format(args.lr))
    logger.info('best acc={:.3f}'.format(best_acc1))

def calculate_metrics(preds, labels, num_classes):
    """
    计算语义分割的评估指标：mIoU, mDice, Pixel Accuracy, Mean Pixel Accuracy, 
    Frequency Weighted IoU, mF1, 95% Hausdorff Distance (HD95)
    :param preds: 模型预测结果，形状为 (batch_size, height, width)，整数类型
    :param labels: 标签，形状为 (batch_size, height, width)，整数类型
    :param num_classes: 类别数
    :return: 各指标的元组 (mIoU, mDice, PA, mPA, FWIoU, mF1, HD95)
    """
    # 初始化指标列表
    ious = []
    class_dices = []  # 每个类别的Dice系数
    class_accuracies = []
    class_f1s = []
    total_correct = 0
    total_pixels = labels.numel()  # 总像素数
    freq_weighted_iou = 0
    hd95_list = []  # 每个类别的95%豪斯多夫距离

    # 转换为numpy数组（方便坐标提取）
    preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    for i in range(num_classes):
        # 1. 计算TP、FP、FN（用于IoU、Dice、F1）
        true_positive = ((preds == i) & (labels == i)).sum().item()
        false_positive = ((preds == i) & (labels != i)).sum().item()
        false_negative = ((preds != i) & (labels == i)).sum().item()

        # 2. 计算IoU
        union = true_positive + false_positive + false_negative
        iou = true_positive / union if union > 0 else 0.0
        ious.append(iou)

        # 3. 计算Dice系数（2*TP / (2*TP + FP + FN)）
        dice_denominator = 2 * true_positive + false_positive + false_negative
        dice = 2 * true_positive / dice_denominator if dice_denominator > 0 else 0.0
        class_dices.append(dice)

        # 4. 类别像素总数（判断该类别是否在标签中存在）
        total_class_pixels = (labels == i).sum().item()

        if total_class_pixels > 0:
            # 类别准确率
            class_acc = true_positive / total_class_pixels
            class_accuracies.append(class_acc)

            # 加权IoU
            freq_weighted_iou += total_class_pixels * iou

            # F1-score（精确率+召回率）
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            class_f1s.append(f1)

            # # 5. 计算95%豪斯多夫距离（HD95）
            # # 提取当前类别的预测和标签的二值掩码
            # pred_mask = (preds_np == i).astype(np.uint8)
            # label_mask = (labels_np == i).astype(np.uint8)

            # # 提取掩码中非零像素的坐标（点集）
            # pred_coords = np.argwhere(pred_mask == 1)  # 形状：(N, 3)，(batch, h, w)坐标
            # label_coords = np.argwhere(label_mask == 1)  # 形状：(M, 3)

            # # 处理空点集情况
            # if len(pred_coords) == 0 and len(label_coords) == 0:
            #     # 两者都为空，距离为0
            #     hd95 = 0.0
            # elif len(pred_coords) == 0 or len(label_coords) == 0:
            #     # 一方为空，距离设为最大可能值（或根据需求调整）
            #     hd95 = float('inf')  # 或图像对角线长度（如sqrt(256^2 + 256^2)）
            # else:
            #     # 计算双向豪斯多夫距离
            #     # 预测到标签的距离：每个预测点到最近标签点的距离
            #     dist_p2l = cdist(pred_coords, label_coords, metric='euclidean').min(axis=1)
            #     # 标签到预测的距离：每个标签点到最近预测点的距离
            #     dist_l2p = cdist(label_coords, pred_coords, metric='euclidean').min(axis=1)
            #     # 合并距离并取95%分位数
            #     all_distances = np.concatenate([dist_p2l, dist_l2p])
            #     hd95 = np.percentile(all_distances, 95)  # 95%分位数

            # hd95_list.append(hd95)

        else:
            class_accuracies.append(0.0)

        # 累计总体正确像素
        total_correct += true_positive

    # 计算最终指标
    mIoU = np.mean([iou for iou in ious if iou > 0]) if ious else 0.0
    mDice = np.mean([dice for dice in class_dices if dice > 0]) if class_dices else 0.0
    PA = total_correct / total_pixels if total_pixels > 0 else 0.0
    mPA = np.mean([acc for acc in class_accuracies if acc > 0]) if class_accuracies else 0.0
    FWIoU = freq_weighted_iou / total_pixels if total_pixels > 0 else 0.0
    mF1 = np.mean(class_f1s) if class_f1s else 0.0
    # HD95 = np.mean(hd95_list) if hd95_list else 0.0  # 平均所有存在类别的HD95

    return mIoU, mPA

def train(train_loader, model, scheduler, criterion,dicefocal, optimizer, epoch, args):

    if args.gpu is not None and args.gpu != 'cpu' and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        # 否则使用 CPU
        device = torch.device("cpu")

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mIOU = AverageMeter('mIOU', ':6.4f')
    mAcc = AverageMeter('mAcc', ':6.4f')
    LR = AverageMeter('learning rate', ':6.5f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time,losses, mIOU , mAcc, LR, data_time],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # 存储每轮的训练损失、验证损失和MIoU
    train_miou = []
    train_mPA = []
    Geo_loss = []

    end = time.time()

    for i, (imageA, imageB, target) in enumerate(train_loader):

        data_time.update(time.time() - end)  # measure data loading time
        if args.gpu is not None:
            imageA = imageA.to(device, non_blocking=True)
            imageB = imageB.to(device, non_blocking=True)

        if args.gpu is not None:
            # target = target.cuda(args.gpu, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target.long()

        out, geo_loss = model(imageA, imageB)

        loss = 1 * criterion(out, target) + 0.5 * dicefocal(out, target) + 0.5 * geo_loss
      
        output = torch.argmax(out, dim=1)
        mean_iou, accuracy = calculate_metrics(output, target, num_classes=args.numclass)
        train_miou.append(mean_iou)
        train_mPA.append(accuracy)
        Geo_loss.append(geo_loss)


        losses.update(loss.item(), imageA.size(0))
        mIOU.update(mean_iou,imageA.size(0))
        mAcc.update(accuracy, imageA.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm为裁剪阈值
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        LR.update(lr,imageA.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    #
        if i % args.print_freq == 0:
            progress.display(i)

    scheduler.step()
    print(sum(Geo_loss)/len(Geo_loss))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mIOU = AverageMeter('mIOU', ':6.2f')
    mAcc = AverageMeter('mAcc', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mIOU,mAcc, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if args.gpu is not None and args.gpu != 'cpu' and torch.cuda.is_available():
        # 有效 GPU 设备（如 args.gpu=0）
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        # 否则使用 CPU
        device = torch.device("cpu")


    with torch.no_grad():
        end = time.time()
        for i, (imageA,imageB,target) in enumerate(val_loader):
            if args.gpu is not None:
                imageA = imageA.to(device, non_blocking=True)
                imageB = imageB.to(device, non_blocking=True)
            if args.gpu is not None:
                # target = target.cuda(args.gpu, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target = target.long()

            out, geo_loss = model(imageA,imageB)
            loss = 1 * criterion(out, target)

            output = torch.argmax(out, dim=1)
            mean_iou, accuracy = calculate_metrics(output, target, num_classes=args.numclass)

            losses.update(loss.item(), imageA.size(0))
            mIOU.update(mean_iou, imageA.size(0))
            mAcc.update(accuracy, imageA.size(0))
        #
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq_val == 0:
                progress.display(i)
        #
        # # TODO: this should also be done with the ProgressMeter
        print(' * mIOU@ {mIOU:.3f} mAcc@ {mAcc:.3f}'.format(mIOU=mIOU.avg, mAcc=mAcc.avg))

    return mIOU.avg


def save_checkpoint(state, is_best, filename):
    if is_best:
        torch.save(state, filename)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()



