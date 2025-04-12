# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from btcv_utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

from utils import config, WeightConvertor, save_hparams, seed_all
from pathlib import Path
from models import setup_model

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")

# 这里是从 checkpoint 中继续训练的位置，SwinUNETR 环境绑定
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument(
    "--logdir",
    default=Path(config.tensorboard_dir) / "normal" / "p_custom_v14_BTCV_offical",
    type=str,
    help="directory to save the tensorboard logs",
)
parser.add_argument(
    "--pretrained_dir",
    default=Path(config.tensorboard_dir) / "pretrain/pretrain_custom_SwinUNETR_v14",
    type=str,
    help="pretrained checkpoint directory",
)
# parser.add_argument("--from_pretrain", action="store_false", help="是否加载预训练权重")
parser.add_argument(
    "--pretrained_model_name",
    default="last_model.pth",
    type=str,
    help="pretrained model name",
)
parser.add_argument(
    "--data_dir",
    default=Path(config.dataset_path) / "raw_dataset/BTCV/RawData",
    type=str,
    help="dataset directory",
)
parser.add_argument(
    "--json_list",
    default=Path(config.dataset_path) / "raw_dataset/BTCV/RawData/dataset.json",
    type=str,
    help="dataset json file",
)
parser.add_argument("--save_checkpoint", action="store_false", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=config.train_epochs, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=config.batch_size, type=int, help="number of batch size")
parser.add_argument(
    "--sw_batch_size", default=config.sw_batch_size, type=int, help="number of sliding window batch size"
)
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=config.val_interval, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=config.vit_embed_dim, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=config.resample_spacing[0], type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=config.resample_spacing[1], type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=config.resample_spacing[2], type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=config.patch_shape[0], type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=config.patch_shape[1], type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=config.patch_shape[2], type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=config.vit_drop_rate, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=config.vit_dropout_path_rate, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=config.lr_warmup_epochs, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_false", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


# def main():
#     args = parser.parse_args()
#     args.amp = not args.noamp
#     args.logdir = "./runs/" + args.logdir
#     if args.distributed:
#         args.ngpus_per_node = torch.cuda.device_count()
#         print("Found total gpus", args.ngpus_per_node)
#         args.world_size = args.ngpus_per_node * args.world_size
#         mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
#     else:
#         main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    # 不能随便修改 rank，有一些指标和权重的保存仅在 rank 0 上启用。
    # else:
    #     args.rank = config.gpu_id

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    args.test_mode = False
    loader = get_loader(args)

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir

    if config.model_name not in ["DAE", "GVSL"]:
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=config.vit_use_checkpoint,
        )
    else:
        model = setup_model()

    if args.resume_ckpt:
        model_dict = WeightConvertor.convert_from_custom_pretrain(
            torch.load(os.path.join(pretrained_dir, args.pretrained_model_name), weights_only=False)
        )
        # model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.swinViT.load_state_dict(model_dict)
        print("Use custom pretrained weights")
    elif ~args.resume_ckpt and config.model_name in ["DAE", "GVSL"]:
        print(f"training from {config.model_name}")
    else:
        print("training from scrach")

    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load("./pretrained_models/model_swinvit.pt")
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    # 最后查看是否存在先前的存档点
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        # 如果开启分布式训练，则将模型用 DDP 包裹
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    # 确认使用哪种优化器
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    # 确认使用哪种调度器
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    # 开始实验
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    # 固定种子
    seed_all(config.all_seed)
    args = parser.parse_args()
    args.amp = not args.noamp
    # args.logdir = "./runs/" + args.logdir
    # 保存 yaml 文件
    save_hparams(args.logdir.name, mode="normal")

    if args.distributed:
        print(f"多GPU训练")
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=config.gpu_id, args=args)

    # main()
