import math
import sys
from typing import Iterable
import torch
import utils.misc as utils
import numpy as np
import torch.distributed as dist
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, roc_curve
import os

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, lr_scheduler = None, max_norm: float = 0, args=None, model_ema=None):
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        images, labels = [sample.to(device) for sample in samples]
        outputs = model(images)
        if args.distributed:
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        loss_dict = model_without_ddp.get_criterion(outputs, labels)
        weight_dict = model_without_ddp.criterion_weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # original backward function
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # lr_scheduler.step()
        if model_ema:
            model_ema.update(model)
            
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


@torch.no_grad()
def gather_together(data):
    world_size = utils.get_world_size()
    if world_size < 2:
        return data
    if not utils.is_dist_avail_and_initialized():
        return data
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


@torch.no_grad()
def evaluate(model, data_loaders, device, args=None, test=False):
    model.eval()
    # for param_name, param in model.module.prompt_learner.named_parameters():
    #     if "ctx" in param_name:
    #         param.requires_grad = True
    # if test:
    #     images = data_loaders.to(device)
    #     outputs = model(images)
    #     print(outputs.softmax(dim=1)[:, 1].flatten().tolist())
    #     return

    test_dataset = []
    test_AP = []
    test_ACC = []
    test_real_ACC = []
    test_fake_ACC = []
    test_auc = []
    test_hter = []
    test_eer_thr = []
    global_y_true = []
    global_y_pred = []

    for data_name, data_loader in data_loaders.items():
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        print_freq = args.print_freq

        y_true, y_pred = [], []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            images, labels = [sample.to(device) for sample in samples]
            
            outputs = model(images)
            
            y_pred.extend(outputs.sigmoid().flatten().tolist())
            y_true.extend(labels.flatten().tolist())
        
        
        world_size = utils.get_world_size()
        
        if world_size < 2:
            merge_y_true = y_true
        else:
            merge_y_true = []
            for data in gather_together(y_true):
                merge_y_true.extend(data)
        
        if world_size < 2:
            merge_y_pred = y_pred
        else:
            merge_y_pred = []
            for data in gather_together(y_pred):
                merge_y_pred.extend(data)
        
        y_true, y_pred = np.array(merge_y_true), np.array(merge_y_pred)
        global_y_true.extend(y_true.tolist())
        global_y_pred.extend(y_pred.tolist())

        # r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        # f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        # acc = accuracy_score(y_true, y_pred > 0.5)
        # ap = average_precision_score(y_true, y_pred)

        # Add auc and hter, modified by shaoshi 
        # auc = roc_auc_score(y_true, y_pred)
        # thr = 0.5
        # label=1 -> fake, label=0 -> live
        # FAR: fake accepted as live; FRR: live rejected as fake
        # far = np.mean(y_pred[y_true == 1] < thr)
        # frr = np.mean(y_pred[y_true == 0] >= thr)
        # hter = (far + frr) / 2

        # === EER threshold ===
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_thr = thresholds[eer_idx]
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > eer_thr)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > eer_thr)
        acc = accuracy_score(y_true, y_pred > eer_thr)
        ap = average_precision_score(y_true, y_pred)

        auc = roc_auc_score(y_true, y_pred)
        # label=1 -> fake, label=0 -> live
        # FAR: fake accepted as live; FRR: live rejected as fake
        far = np.mean(y_pred[y_true == 1] < eer_thr)
        frr = np.mean(y_pred[y_true == 0] >= eer_thr)
        hter = (far + frr) / 2

        test_dataset.append(data_name)
        test_AP.append(ap)
        test_ACC.append(acc)
        test_real_ACC.append(r_acc)
        test_fake_ACC.append(f_acc)
        test_auc.append(auc)
        test_hter.append(hter)
        test_eer_thr.append(eer_thr)

        # print("({}) acc: {:.2f}; auc: {:.2f}; hter: {:.2f};".format(data_name, acc * 100, auc * 100, hter * 100))
        print("({}) acc: {:.2f}; auc: {:.2f}; hter: {:.2f}; eer_thr: {:.4f};".format(
            data_name, acc * 100, auc * 100, hter * 100, eer_thr))

    # === Global EER threshold (all validation samples) ===
    global_y_true = np.array(global_y_true)
    global_y_pred = np.array(global_y_pred)
    global_fpr, global_tpr, global_thresholds = roc_curve(global_y_true, global_y_pred)
    global_fnr = 1 - global_tpr
    global_eer_idx = np.nanargmin(np.abs(global_fpr - global_fnr))
    global_eer_thr = global_thresholds[global_eer_idx]

    output_strs = []
    for idx, [name, acc, auc, hter, eer_thr] in enumerate(
        zip(
            test_dataset + ["mean"],
            test_ACC + [np.mean(test_ACC)],
            test_auc + [np.mean(test_auc)],
            test_hter + [np.mean(test_hter)],
            test_eer_thr + [np.mean(test_eer_thr)],
        )
    ):
        output_str = "({} {:10}) acc: {:.2f}; auc: {:.2f}; hter: {:.2f}; eer_thr: {:.4f};".format(
            idx, name, acc * 100, auc * 100, hter * 100, eer_thr
        )
        output_strs.append(output_str)
        print(output_str)
    
    return "; ".join(output_strs), np.mean(test_AP), np.mean(test_ACC), np.mean(test_auc), np.mean(test_hter), global_eer_thr
