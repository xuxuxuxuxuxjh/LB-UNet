import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer):

    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets, points = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        points = points.cuda(non_blocking=True).float()

        gt_pre, key_points, out = model(images)
        loss = criterion(gt_pre, key_points, out, targets, points)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    loss_list = []
    total_miou = 0.0
    total = 0
    gt_list = []
    pred_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, key_points, out = model(img)

            gts = msk.squeeze(1).cpu().detach().numpy()
            preds = out.squeeze(1).cpu().detach().numpy()
            gt_list.append(gts)
            pred_list.append(preds)
            preds = np.array(preds).reshape(-1)
            gts = np.array(gts).reshape(-1)

            y_pre = np.where(preds>=config.threshold, 1, 0)
            y_true = np.where(gts>=0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)
            
            total_miou += miou
            total += 1
            
    total_miou = total_miou / total
    pred_list = np.array(pred_list).reshape(-1)
    gt_list = np.array(gt_list).reshape(-1)

    y_pre = np.where(pred_list>=0.5, 1, 0)
    y_true = np.where(gt_list>=0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {total_miou}, f1_or_dsc: {f1_or_dsc}'
    print(log_info)
    logger.info(log_info)
    return - (total_miou + f1_or_dsc)


def test_one_epoch(test_loader, model, criterion, logger, config, path, test_data_name=None):
    model.eval()
    gt_list = []
    pred_list = []
    total_miou = 0.0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, key_points, out = model(img)

            msk = msk.squeeze(1).cpu().detach().numpy()
            out = out.squeeze(1).cpu().detach().numpy()

            gt_list.append(msk)
            pred_list.append(out)
            
            y_pre = np.where(out>=config.threshold, 1, 0)
            y_true = np.where(msk>=0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)

            total_miou += miou
            total += 1

            # if i % config.save_interval == 0:
                # kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12 = key_points
                # gt1, gt2, gt3, gt4, gt5 = gt_pre
                # save_imgs(img, msk, out, key_points, gt_pre, i, config.work_dir + 'outputs/' + 'ISIC2017' + '/', config.datasets, config.threshold, test_data_name=test_data_name)

        total_miou = total_miou / total

        pred_list = np.array(pred_list).reshape(-1)
        gt_list = np.array(gt_list).reshape(-1)

        y_pre = np.where(pred_list>=0.5, 1, 0)
        y_true = np.where(gt_list>=0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        
        log_info = f'test of best model, miou: {total_miou}, f1_or_dsc: {f1_or_dsc}'
        print(log_info)
        logger.info(log_info)