import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import copy
from models.head import iBOTHead
# from evaluation.unsupervised.unsup_cls import eval_pred
from dataset.datasets_us308k_flexible import FlexibleUltrasoundImageDataset
import wandb
from vmamba_models.dino_vmamba import dinov2_vmamba_small, Backbone_DINOv2_VSSM_2


def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vmamba_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'vmamba_small', 'vmamba_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
    parser.add_argument('--enable_wandb', default=False, type=utils.bool_flag)
    
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=151, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='.', type=str,
        help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    #     help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--train_num", default=1, type=int)
    parser.add_argument("--debug", default=False, type=utils.bool_flag)
    parser.add_argument("--pretrained_vmamba", default=False, type=utils.bool_flag)
    parser.add_argument("--mask_model", default="random", type=str)
    parser.add_argument("--masking_ratio", default=0.60, type=float)
    parser.add_argument("--local_rec_loss", default=True, type=utils.bool_flag)
    parser.add_argument("--global_rec_loss", default=True, type=utils.bool_flag)
    parser.add_argument("--wandb_name", default='attenmask', type=str)
    parser.add_argument("--student_feedback", default=False, type=utils.bool_flag)
    parser.add_argument("--debug_epoch", default=20, type=int)
    # Adaptive weighting parameters
    parser.add_argument("--adaptive_weighting", default=False, type=utils.bool_flag, help="Enable adaptive weighting between teacher attention and student reconstruction importance")
    parser.add_argument("--alpha_init", default=0.2, type=float, help="Initial alpha value for adaptive weighting (0=student only, 1=teacher only)")
    parser.add_argument("--alpha_final", default=0.5, type=float, help="Final alpha value for adaptive weighting")
    parser.add_argument("--alpha_schedule", default="cosine", type=str, choices=["cosine", "linear", "constant"], help="Schedule for alpha parameter: cosine, linear, or constant")
    return parser

def train_ibot(args):
    if args.enable_wandb:
        wandb.init(project=".", entity=".",  name=f'train_USDION_{args.arch}_{args.train_num}_{args.wandb_name}')
    
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if 'swin' or 'vmamba' in args.arch else args.patch_size
    
    dataset = FlexibleUltrasoundImageDataset(
        root=args.data_path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio, 
        pred_ratio_var=args.pred_ratio_var, 
        pred_aspect_ratio=(0.3, 1/0.3), 
        pred_shape=args.pred_shape, 
        pred_start_epoch=args.pred_start_epoch)
    
    print(f"Data loaded: there are {len(dataset)} images.")
    # dataset = ImageFolderMask(
    #     args.data_path, 
    #     transform=transform,
    #     patch_size=pred_size,
    #     pred_ratio=args.pred_ratio,
    #     pred_ratio_var=args.pred_ratio_var,
    #     pred_aspect_ratio=(0.3, 1/0.3),
    #     pred_shape=args.pred_shape,
    #     pred_start_epoch=args.pred_start_epoch)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif args.arch == 'vmamba_small':
        if args.pretrained_vmamba:
            student = Backbone_DINOv2_VSSM_2(pretrained='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth', 
                                           masked_im_modeling=args.use_masked_im_modeling)
            teacher = Backbone_DINOv2_VSSM_2(pretrained='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth', 
                                           masked_im_modeling=args.use_masked_im_modeling)
            embed_dim = student.dims[-1]
            print(f"Using embed_dim: {embed_dim}")
        else:
            student = dinov2_vmamba_small(
                patch_size=args.patch_size, 
                return_all_tokens=True,
                masked_im_modeling=args.use_masked_im_modeling
                )
            teacher = dinov2_vmamba_small(
                patch_size=args.patch_size, 
                return_all_tokens=True,
                masked_im_modeling=args.use_masked_im_modeling
                )
            embed_dim = student.dims[-1]
    elif args.arch == 'vmamba_base': #depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,  SSM_RATIO: 2.0
        if args.pretrained_vmamba:
            student = Backbone_DINOv2_VSSM_2(depths=[2,2,15,2], dims=128, drop_path_rate=0.6, ssm_ratio=2.0, pretrained='./pretrained/vmamba/vssm_base_0229_ckpt_epoch_237.pth', 
                                            masked_im_modeling=args.use_masked_im_modeling)
            teacher = Backbone_DINOv2_VSSM_2(depths=[2,2,15,2], dims=128, drop_path_rate=0.6, ssm_ratio=2.0, pretrained='./pretrained/vmamba/vssm_base_0229_ckpt_epoch_237.pth', 
                                            masked_im_modeling=args.use_masked_im_modeling)
            embed_dim = student.dims[-1]
            print(f"Using embed_dim: {embed_dim}")
        else:
            student = Backbone_DINOv2_VSSM_2(depths=[2,2,15,2], dims=128, drop_path_rate=0.6, ssm_ratio=2.0, 
                                    masked_im_modeling=args.use_masked_im_modeling)
            teacher = Backbone_DINOv2_VSSM_2(depths=[2,2,15,2], dims=128, drop_path_rate=0.6, ssm_ratio=2.0, 
                                    masked_im_modeling=args.use_masked_im_modeling)
            embed_dim = student.dims[-1]
            print(f"Using embed_dim: {embed_dim}")
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True,  static_graph=True) if \
        ('swin' in args.arch or 'vmamba' in args.arch) else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True, static_graph=True) if \
    ('swin' in args.arch or 'vmamba' in args.arch) else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
 
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    
    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    # if utils.is_main_process(): # Tensorboard configuration
    #     local_runs = os.path.join(args.output_dir, 'tf_logs')
    #     writer = SummaryWriter(logdir=local_runs)
    
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        # args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.lr * (args.batch_size_per_gpu/args.batch_size_per_gpu),
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    # if args.debug:
    #     start_epoch = 0
    # else:
    start_epoch = to_restore["epoch"]
    print(f"start_epoch: {start_epoch}")
    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        # if fp16_scaler is not None:
        #     save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        # utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            # utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        # if utils.is_main_process():
        log_path = os.path.join(args.output_dir, f"log_{args.arch}.txt")
        print(f"Writing logs to {log_path}")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_stats) + "\n")
            # for k, v in train_stats.items():
            #     writer.add_scalar(k, v, epoch)
        
        if args.enable_wandb:
            log_dict = {
                'train_loss': train_stats['loss'],
                'train_lr': train_stats['lr'],
                'train_wd': train_stats['wd'],
                'train_loss_cls': train_stats['cls'],
                'train_loss_patch': train_stats['patch']
            }
            
            if args.global_rec_loss and not args.local_rec_loss:
                log_dict['train_loss_rec'] = train_stats['global_rec_loss']
            elif args.global_rec_loss and args.local_rec_loss:
                log_dict['train_loss_rec'] = train_stats['global_rec_loss']
                log_dict['train_loss_local_rec'] = train_stats['local_rec_loss']
            
            if args.student_feedback and args.adaptive_weighting and 'alpha' in train_stats:
                log_dict['alpha'] = train_stats['alpha']
            
            wandb.log(log_dict)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    # Initialize reconstruction importance storage for adaptive weighting
    if not hasattr(train_one_epoch, 'prev_recon_importance'):
        train_one_epoch.prev_recon_importance = None
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    mask_prev = None 
    current_alpha = 0.0  # Track current alpha value
    for it, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it_count = it
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]        
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output, teacher_attn = teacher(images[:args.global_crops_number], return_attn=True)
            
            if args.student_feedback:
                m_t = 0.1 + (epoch / args.epochs) * (0.9 - 0.1)
                
                if args.adaptive_weighting:
                    if args.alpha_schedule == "cosine":
                        alpha = args.alpha_init + (args.alpha_final - args.alpha_init) * (1 - math.cos(math.pi * epoch / args.epochs)) / 2
                    elif args.alpha_schedule == "linear":
                        alpha = args.alpha_init + (args.alpha_final - args.alpha_init) * (epoch / args.epochs)
                else:
                    alpha = args.alpha_init # constant
                    
                current_alpha = alpha  # Store current alpha for logging  
                if not hasattr(train_one_epoch, 'prev_recon_importance') or train_one_epoch.prev_recon_importance is None:
                    _, mask_generated = utils.GlobalAttGuidedMask_2(teacher_attn, masking_ratio=args.masking_ratio, b=args.batch_size_per_gpu*2, T=1, threshold=m_t, patch_size=args.patch_size*8)
                else:
                    # Process teacher attention A_t: [B, 49, 49] -> [B, 49]
                    A_t = teacher_attn.mean(dim=1)  # [B, 49]
                    # Process student reconstruction importance L_s: [B, 768, 49] -> [B, 49]
                    L_s = train_one_epoch.prev_recon_importance.mean(dim=1)  # [B, 49]
                    
                    A_t = (A_t - A_t.min(dim=1, keepdim=True)[0]) / (A_t.max(dim=1, keepdim=True)[0] - A_t.min(dim=1, keepdim=True)[0] + 1e-8)
                    L_s = (L_s - L_s.min(dim=1, keepdim=True)[0]) / (L_s.max(dim=1, keepdim=True)[0] - L_s.min(dim=1, keepdim=True)[0] + 1e-8)
                    
                    # masking_score = (1-alpha) * A_t + alpha * L_s  # [B, 49]
                    masking_score = alpha * A_t + (1-alpha) * L_s  # [B, 49]
                    masking_score_expanded = masking_score.unsqueeze(1).expand(-1, masking_score.size(-1), -1)  # [B, 49, 49]
                    _, mask_generated = utils.GlobalAttGuidedMask_2(masking_score_expanded, masking_ratio=args.masking_ratio, b=args.batch_size_per_gpu*2, T=1, threshold=m_t, patch_size=args.patch_size*8)
                        
                mask_generated = mask_generated.bool()
                mask_generated_list = mask_generated.chunk(args.global_crops_number)
                 
                if args.global_rec_loss:
                    student_output, global_rec_loss, recon_importance = student(images[:args.global_crops_number], mask=mask_generated_list, return_rec_loss=True)
                    train_one_epoch.prev_recon_importance = recon_importance.detach()
                else:
                    student_output = student(images[:args.global_crops_number], mask=mask_generated_list)
                    train_one_epoch.prev_recon_importance = None
            else:    
                if args.mask_model == "random": 
                    if args.global_rec_loss:
                        student_output, global_rec_loss, _ = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number], return_rec_loss=True)
                    else:
                        student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])
                elif args.mask_model == "atten_guided":
                    m_t = 0.001 + (epoch / args.epochs) * (0.9 - 0.001)
                    mask_indices, mask_generated = utils.GlobalAttGuidedMask_2(teacher_attn, masking_ratio=args.masking_ratio, b=args.batch_size_per_gpu*2, T=1, threshold=m_t, patch_size=args.patch_size*8)
                    mask_generated = mask_generated.bool()
                    mask_generated_list = mask_generated.chunk(args.global_crops_number)
                    if args.global_rec_loss:
                        student_output, global_rec_loss, _ = student(images[:args.global_crops_number], mask=mask_generated_list, return_rec_loss=True)
                    else:
                        student_output = student(images[:args.global_crops_number], mask=mask_generated_list)

            # get local views
            student.module.backbone.masked_im_modeling = False
            
            if args.local_rec_loss:
                student_local, local_rec_loss = student(images[args.global_crops_number:], mask=masks[args.global_crops_number:], return_rec_loss=True, return_recon_importance=False) if len(images) > args.global_crops_number else None
                student_local_cls = student_local[0]
            else:
                student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None

            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            if args.mask_model == "random": 
                all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            elif args.mask_model == "atten_guided":
                all_loss = ibot_loss(student_output, teacher_output, student_local_cls, mask_generated_list, epoch)

            loss = all_loss.pop('loss')

            if args.global_rec_loss:
                loss += global_rec_loss
            
            if args.local_rec_loss:
                loss += local_rec_loss
                
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            
        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        if args.global_rec_loss:
            metric_logger.update(global_rec_loss=global_rec_loss.item())
        if args.local_rec_loss:
            metric_logger.update(local_rec_loss=local_rec_loss.item())
        if args.student_feedback and args.adaptive_weighting:
            metric_logger.update(alpha=current_alpha)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        # metric_logger.update(acc=acc)

    # pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    # real_labels = torch.cat(real_labels).cpu().detach().numpy()
    # nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=0.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    # loss2 = F.cosine_similarity(teacher_patch_c[q], student_patch_c[v], dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.227, 0.227, 0.227), (0.191, 0.191, 0.191)), # USMSK
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            # transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            # transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
