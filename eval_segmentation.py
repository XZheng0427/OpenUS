import os
import argparse
import json
import copy
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from torchvision import transforms as pth_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import models
from vmamba_models.dino_vmamba import (
    dinov2_vmamba_small,
    Backbone_DINOv2_VSSM_2,
)
from vmamba_models.MambaDecoder import MambaDecoder
from torchvision import transforms as pth_transforms
from dataset.dataset_busbra import BUSBRADataset
from dataset.dataset_tn3k import TN3KDataset
from dataset.transforms import get_transforms

class SegmentationHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, tokens, orig_hw):
        # tokens: [B, N, D] where N = h*w patches
        B, N, D = tokens.shape
        h = w = int(math.sqrt(N))
        feat = tokens.permute(0,2,1).view(B, D, h, w)
        logits = self.conv1x1(feat)  # [B, C, h, w]
        # upsample to original resolution
        return F.interpolate(logits, size=orig_hw, mode='bilinear', align_corners=False)


class MambaDecoderHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.decoder = MambaDecoder(num_classes=num_classes)

    def forward(self, encoder_out, orig_hw):
        logits = self.decoder.forward(encoder_out)
  
        return F.interpolate(logits, size=orig_hw, mode='bilinear', align_corners=False)


def compute_iou(pred, target, eps=1e-6):
    pred_flat   = pred.view(-1).byte()
    target_flat = target.view(-1).byte()

    # binary masks for foreground
    pred_fg   = pred_flat == 1
    target_fg = target_flat == 1

    # compute intersection and union
    intersection = (pred_fg & target_fg).sum().float()
    union = (pred_fg | target_fg).sum().float()

    # handle empty union (no foreground in either pred or target) as IoU=1
    if union.item() == 0:
        return 1.0

    return ((intersection + eps) / (union + eps)).item()


def compute_dice(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    # flatten
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    # binary masks for foreground class
    pred_fg   = pred_flat == 1
    target_fg = target_flat == 1

    intersection = (pred_fg & target_fg).sum().float()
    dice = (2. * intersection + eps) / (pred_fg.sum().float() + target_fg.sum().float() + eps)
    return dice.item()


def train_seg(model, head, optimizer, loader, epoch, criterion):
    model.eval()
    head.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    for imgs, masks, mask_filenames in metric_logger.log_every(loader, 20, header):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        
        with torch.no_grad():
            if args.arch == 'vmamba_small':
                output = model(imgs)
                # output = output[:, 1:]
        
        logits = head(output, orig_hw=imgs.shape[2:])
        loss = criterion(logits, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_seg(val_loader, model, head, criterion, args):
    model.eval()
    head.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    total_iou = 0.0
    total_dice = 0.0
    count = 0
    for imgs, masks, mask_filenames in metric_logger.log_every(val_loader, 20, 'Test:'):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        
        with torch.no_grad():
            if args.arch == 'vmamba_small':
                output = model(imgs)
                # output = output[:, 1:]
        
        logits = head(output, orig_hw=imgs.shape[2:])
        
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)
        batch_iou = compute_iou(preds, masks)
        batch_dice = compute_dice(preds, masks)
        total_iou += batch_iou
        total_dice += batch_dice
        count += 1
        
        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
    return {'loss': metric_logger.meters['loss'].global_avg,
            'miou': total_iou / max(count, 1),
            'dice': total_dice / max(count, 1)}


def eval_seg(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k,v in sorted(vars(args).items())))
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    train_transform, val_transform = get_transforms(img_size=224)
    
    if args.dataset_name == 'BUSBRA':
        organized_data_dir = "."
        fold = '0'  # Use fold 1
        train_ds = BUSBRADataset(organized_data_dir, fold, split='train', transform=train_transform)
        val_ds = BUSBRADataset(organized_data_dir, fold, split='validation', transform=val_transform)
    elif args.dataset_name == 'TN3K':
        train_ds = TN3KDataset(image_dir=args.data_root, mask_dir=args.data_root2, json_file=args.json_file, split='train', transform=train_transform)
        val_ds   = TN3KDataset(image_dir=args.data_root, mask_dir=args.data_root2, json_file=args.json_file,   split='val',   transform=val_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size_per_gpu, sampler=train_sampler,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader    = DataLoader(val_ds,   batch_size=args.batch_size_per_gpu, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    print(f"Data loaded: {len(train_ds)} train, {len(val_ds)} val samples.")

    # ============ building network ... ============
    if 'swin' in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](window_size=args.window_size,
                                            patch_size=args.patch_size,
                                            num_classes=0)
        embed_dim = model.num_features
    elif args.arch == 'vmamba_small':
        if args.pretrained_vmamba:
            model = Backbone_DINOv2_VSSM_2(pretrained='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth',
                                          seg_head=True)
        else:
            model = dinov2_vmamba_small(patch_size=args.patch_size,
                                        return_all_tokens=True,
                                        masked_im_modeling=False)
        embed_dim = model.dims[-1]
    else:
        model = models.__dict__[args.arch](patch_size=args.patch_size,
                                           num_classes=0,
                                           use_mean_pooling=args.avgpool_patchtokens==1)
        embed_dim = model.embed_dim
    model.cuda()
    print(f"Backbone {args.arch} built.")

    # load pretrained backbone
    if args.arch == 'vmamba_small':
        if os.path.isfile(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
            print("Keys in checkpoint:", checkpoint.keys())

            if args.checkpoint_key is not None and args.checkpoint_key in checkpoint:
                print(f"Taking key {args.checkpoint_key} in checkpoint")
                state_dict = checkpoint[args.checkpoint_key]
                print("State dict keys example (first 5):", list(state_dict.keys())[:5])
                
                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_key = k[9:]  # Remove 'backbone.'
                        clean_state_dict[new_key] = v
                
                try:
                    msg = model.load_state_dict(clean_state_dict, strict=False)
                    print(f"Successfully loaded pretrained weights with msg: {msg}")
                except Exception as e:
                    print(f"Error during loading: {e}")
                    print("WARNING: Could not load pretrained weights. Training with random init.")
            else:
                print(f"No key '{args.checkpoint_key}' found. Available keys:", list(checkpoint.keys()))
                print("WARNING: Could not load pretrained weights. Training with random init.")
        else:
            print(f"No pretrained weights found at {args.pretrained_weights}")
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights,
                                      args.checkpoint_key, args.arch, args.patch_size)

    # freeze backbone
    for p in model.parameters(): p.requires_grad = False

    # segmentation head
    # head = SegmentationHead(embed_dim, args.num_classes).cuda() #simple head
    head = MambaDecoderHead(num_classes=args.num_classes).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    # optionally resume
    to_restore = {"epoch": 0, "best_miou": 0., "best_dice": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(os.path.join(args.output_dir, args.load_from),
                                      run_variables=to_restore,
                                      state_dict=head,
                                      optimizer=optimizer,
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]
    best_miou   = to_restore["best_miou"]
    best_dice   = to_restore["best_dice"]
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_seg(model, head, optimizer, train_loader, epoch, criterion)
        scheduler.step()

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            metrics = validate_seg(val_loader, model, head, criterion, args)
            val_loss, miou, dice = metrics['loss'], metrics['miou'], metrics['dice']
            print(f"Epoch {epoch} Val Loss {val_loss:.4f} mIoU {miou:.4f} Dice {dice:.4f}")
            
            # Check if validation performance improved
            improved = False
            if miou > best_miou:
                best_miou = miou
                improved = True
            if dice > best_dice:
                best_dice = dice
                improved = True
            
            # log
            log_stats = {**{'epoch': epoch, 'train_loss': train_loss['loss'] if isinstance(train_loss, dict) else train_loss},
                         **{'val_loss': float(val_loss), 'miou': float(miou), 'dice': float(dice), 'best_miou': float(best_miou), 'best_dice': float(best_dice)}}
            path_txt = os.path.join(args.output_dir, f"log_{args.log_name}.txt")
            with open(path_txt, 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
            
            if improved:
                save_dict = {
                    'epoch': epoch+1,
                    'state_dict': head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_miou': best_miou,
                    'best_dice': best_dice
                }
                torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_seg_best.pth"))
                print(f"Saved best checkpoint at epoch {epoch} with mIoU: {miou:.4f}, Dice: {dice:.4f}")
            
            if (epoch + 1) % 10 == 0:
                save_dict = {
                    'epoch': epoch+1,
                    'state_dict': head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_miou': best_miou,
                    'best_dice': best_dice
                }
                torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_seg_epoch_{epoch+1}.pth"))
                print(f"Saved checkpoint at epoch {epoch+1}")
            
            print(f"Max mIoU so far: {best_miou:.4f}")
            print(f"Max Dice so far: {best_dice:.4f}")
    print(f"Segmentation training completed. Best mIoU: {best_miou:.4f}, Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with segmentation on medical images')
    parser.add_argument('--train_csv', default='./BUSBRA/BUSBRA/5-fold-cv.csv', type=str)
    parser.add_argument('--val_csv', default='./BUSBRA/BUSBRA/5-fold-cv.csv', type=str)
    parser.add_argument('--json_file', default='./tn3k/tn3k-trainval-fold0.json', type=str)
    parser.add_argument('--data_root', default='./BUSBRA/BUSBRA/Images/', type=str)
    parser.add_argument('--data_root2', default='./BUSBRA/BUSBRA/Masks/', type=str)
    parser.add_argument('--dataset_name', default='BUSBRA', type=str)
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0,1,2], type=int)
    parser.add_argument('--arch', default='vmamba_small', type=str,
                        choices=['vit_tiny','vit_small','vit_base','vit_large',
                                 'swin_tiny','swin_small','swin_base','swin_large',
                                 'resnet50','resnet101','vmamba_small'])
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--window_size', default=7, type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--output_dir', default='.', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--ignore_index', default=255, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_from', default=None, type=str)
    parser.add_argument('--log_name', default='seg', type=str)
    parser.add_argument("--pretrained_vmamba", default=False, type=utils.bool_flag)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoint_key.split(','):
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = ck
        print(f"Starting segmentation eval for key: {ck}")
        eval_seg(args_copy)