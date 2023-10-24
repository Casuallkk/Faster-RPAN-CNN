import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init, ModelEMA)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (seed_everything,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    seed = 11
    train_gpu = [0, ]
    fp16 = False
    classes_path = 'model_data/annotation_classes.txt'
    model_path = 'model_data/voc_weights_resnet.pth'
    input_shape = [600, 600]
    backbone = "resnet50"
    pretrained = True
    # mosaic增强
    mosaic = True
    mosaic_prob = 0.5

    # mixup增强
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7

    #   anchors_size用于设定先验框的大小，每个特征点均存在9个先验框。
    #   anchors_size每个数对应3个先验框。
    #   当anchors_size = [8, 16, 32]的时候，生成的先验框宽高约为：
    #   [90, 180] ; [180, 360]; [360, 720]; [128, 128]; 
    #   [256, 256]; [512, 512]; [180, 90] ; [360, 180]; 
    #   [720, 360]; 详情查看anchors.py
    #   如果想要检测小物体，可以减小anchors_size靠前的数。
    #   比如设置anchors_size = [4, 16, 32]
    anchors_size = [8, 16, 32]
    # 标签平滑
    label_smoothing = 0

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 4
    Freeze_Train = True

    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 10
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10
    num_workers = 2
    train_annotation_path = 'model_data/train.txt'
    val_annotation_path = 'model_data/val.txt'

    class_names, num_classes = FRCNNDataset.get_classes(classes_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Number of devices: {}'.format(ngpus_per_node))
    seed_everything(seed)

    model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    #   权值平滑
    ema = ModelEMA(model_train)

    #   读取数据集对应的txt
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        #   冻结bn层
        model.freeze_bn()

        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #   判断当前batch_size，自适应调整学习率
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        #   判断每一个世代的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        if ema:
            ema.updates = epoch_step * Init_Epoch
        # 构建数据集加载器
        train_dataset = FRCNNDataset(train_lines, input_shape,
                                     train=True, mosaic=mosaic,
                                     mixup=mixup, mosaic_prob=mosaic_prob,
                                     mixup_prob=mixup_prob,
                                     special_aug_ratio=special_aug_ratio)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False,
                                   mosaic=False, mixup=False,
                                   mosaic_prob=0, mixup_prob=0,
                                   special_aug_ratio=0)

        train_data = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate,
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        val_data = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, collate_fn=frcnn_dataset_collate,
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util = FasterRCNNTrainer(model_train, optimizer)
        #   记录eval的map曲线
        eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda,
                                     eval_flag=eval_flag, period=eval_period)

        #   开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #   判断当前batch_size，自适应调整学习率
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #   获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.extractor.parameters():
                    param.requires_grad = True
                #   冻结bn层
                model.freeze_bn()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                train_data = DataLoader(train_dataset, shuffle=True,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        drop_last=True, collate_fn=frcnn_dataset_collate,
                                        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                val_data = DataLoader(val_dataset, shuffle=True,
                                      batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True, collate_fn=frcnn_dataset_collate,
                                      worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model, model_train, train_util, ema, loss_history, eval_callback,
                          optimizer, epoch, epoch_step, epoch_step_val,
                          train_data, val_data, UnFreeze_Epoch, Cuda, fp16, scaler,
                          save_period, save_dir)

        loss_history.writer.close()
