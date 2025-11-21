#!/usr/bin/env python3
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
from dataset.dataset_tn3k import TN3KDataset, TN3KTestDataset
from dataset.transforms import get_transforms


class MambaDecoderHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.decoder = MambaDecoder(num_classes=num_classes)

    def forward(self, encoder_out, orig_hw):
        logits = self.decoder.forward(encoder_out)
        return F.interpolate(logits, size=orig_hw, mode='bilinear', align_corners=False)


def save_predicted_masks(preds, save_dir, mask_filenames, batch_idx, sample_offset=0, prefix="", max_samples=8):
    batch_size = preds.shape[0]
    num_samples = min(batch_size, max_samples)
    
    for i in range(num_samples):
        try:
            # Convert prediction to numpy and scale to 0-255 for saving as image
            pred_mask = preds[i].cpu().numpy()
            pred_mask = (pred_mask * 255).astype(np.uint8)  # Convert from 0-1 to 0-255
            
            # Create PIL Image and save
            mask_image = Image.fromarray(pred_mask, mode='L')  # 'L' for grayscale
            
            # Use original mask filename (remove extension and add pred_ prefix)
            if i < len(mask_filenames):
                original_filename = mask_filenames[i]
                # Remove extension and add pred_ prefix
                filename_without_ext = os.path.splitext(original_filename)[0]
                filename = f"{filename_without_ext}.png"
            else:
                # Fallback to generic naming if filename not available
                global_sample_idx = sample_offset + i
                filename = f"pred_mask_{global_sample_idx:04d}.png"
            
            save_path = os.path.join(save_dir, filename)
            mask_image.save(save_path)
            
        except Exception as e:
            print(f"Warning: Failed to save mask for batch {batch_idx}, sample {i}: {e}")
            continue


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


@torch.no_grad()
def test_seg(test_loader, model, head, criterion, args, save_predictions=False):
    model.eval()
    head.eval()
    
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    total_samples = 0  # Count samples, not batches
    
    all_predictions = []
    all_targets = []
    
    mask_dir = None
    if args.save_masks:
        mask_dir = os.path.join(args.output_dir, f"predicted_masks_{args.checkpoint_key}")
        os.makedirs(mask_dir, exist_ok=True)
        print(f"Predicted masks will be saved to: {mask_dir}")
        print(f"NOTE: ALL test samples will have masks saved (estimated {len(test_loader) * args.batch_size_per_gpu} images)")
    
    print("Starting test evaluation...")
    
    for i, (imgs, masks, mask_filenames) in enumerate(test_loader):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        
        if args.arch == 'vmamba_small':
            output = model(imgs)
            # output = output[:, 1:]
        
        logits = head(output, orig_hw=imgs.shape[2:])
        
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)
        
        batch_size = preds.shape[0]
        for j in range(batch_size):
            sample_iou = compute_iou(preds[j:j+1], masks[j:j+1])
            sample_dice = compute_dice(preds[j:j+1], masks[j:j+1])
            
            total_iou += sample_iou
            total_dice += sample_dice
            total_samples += 1
        
        total_loss += loss.item() * batch_size
        
        if save_predictions:
            all_predictions.append(preds.cpu())
            all_targets.append(masks.cpu())
        
        if args.save_masks and mask_dir is not None:
            try:
                save_predicted_masks(
                    preds.cpu(), 
                    mask_dir, 
                    mask_filenames,  
                    i, 
                    sample_offset=total_samples - batch_size,  
                    prefix=f"{args.checkpoint_key}_",
                    max_samples=preds.shape[0]  
                )
                
            except Exception as e:
                print(f"Warning: Failed to save masks for batch {i}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_loader)} batches, "
                  f"Running IoU: {total_iou/total_samples:.4f}, "
                  f"Running Dice: {total_dice/total_samples:.4f}")
    
    avg_iou = total_iou / max(total_samples, 1)
    avg_dice = total_dice / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    
    print(f"\n=== Test Results ===")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test mIoU: {avg_iou:.4f}")
    print(f"Test Dice: {avg_dice:.4f}")
    print(f"Total samples: {total_samples}")
    
    if save_predictions and len(all_predictions) > 0:
        pred_save_path = os.path.join(args.output_dir, f"test_predictions_{args.checkpoint_key}.pt")
        torch.save({
            'predictions': torch.cat(all_predictions, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'metrics': {'loss': avg_loss, 'miou': avg_iou, 'dice': avg_dice}
        }, pred_save_path)
        print(f"Predictions saved to: {pred_save_path}")
    
    if args.save_masks and mask_dir is not None:
        print(f"Predicted masks saved to: {mask_dir}")
    
    return {'loss': avg_loss, 'miou': avg_iou, 'dice': avg_dice}


def load_model(args):
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
    return model, embed_dim


def load_pretrained_weights(model, args):
    if args.arch == 'vmamba_small':
        if os.path.isfile(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
            
            if args.checkpoint_key is not None and args.checkpoint_key in checkpoint:
                print(f"Taking key {args.checkpoint_key} in checkpoint")
                state_dict = checkpoint[args.checkpoint_key]
                
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
                    print("WARNING: Could not load pretrained weights.")
            else:
                print(f"No key '{args.checkpoint_key}' found. Available keys:", list(checkpoint.keys()))
        else:
            print(f"No pretrained weights found at {args.pretrained_weights}")
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights,
                                      args.checkpoint_key, args.arch, args.patch_size)


def load_complete_checkpoint(model, head, args):
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_seg_{args.cpk_name}.pth")
    if os.path.isfile(checkpoint_path):
        print(f"Loading complete model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if 'model_state_dict' in checkpoint and 'head_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            head.load_state_dict(checkpoint['head_state_dict'])
            print("Loaded complete model (backbone + head) from checkpoint")
        elif 'complete_model_state_dict' in checkpoint:
            complete_state_dict = checkpoint['complete_model_state_dict']
            
            backbone_state_dict = {}
            head_state_dict = {}
            
            for key, value in complete_state_dict.items():
                if key.startswith('backbone.'):
                    backbone_key = key[9:]  # Remove 'backbone.' prefix
                    backbone_state_dict[backbone_key] = value
                elif key.startswith('head.') or key.startswith('decoder.'):
                    head_key = key.replace('head.', '').replace('decoder.', 'decoder.')
                    head_state_dict[head_key] = value
            
            model.load_state_dict(backbone_state_dict, strict=False)
            head.load_state_dict(head_state_dict, strict=False)
            print("Loaded complete model from unified state dict")
        else:
            head.load_state_dict(checkpoint['state_dict'])
            print("Warning: Checkpoint only contains head weights, backbone uses pretrained weights")
            if args.pretrained_weights:
                load_pretrained_weights(model, args)
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"Best training mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
        print(f"Best training Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        return checkpoint
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    
def load_test_data(args):
    _, test_transform = get_transforms(img_size=224)

    if args.dataset_name == 'BUSBRA':
        organized_data_dir = "./US_DownStreamTask_datasets/BUSBRA/BUSBRA/organized_data"
        fold = '1_60p'  # Use fold 1
        test_ds = BUSBRADataset(organized_data_dir, fold, split='test', transform=test_transform)
    elif args.dataset_name == 'TN3K':
        test_ds = TN3KTestDataset(test_image_dir=args.data_root, test_mask_dir=args.data_root2, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size_per_gpu, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Test data loaded: {len(test_ds)} samples.")
    return test_loader, test_ds


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser('Test segmentation model on medical images')
    
    # Data arguments
    parser.add_argument('--test_csv', default='./BUSBRA/BUSBRA/5-fold-cv.csv', type=str)
    parser.add_argument('--data_root', default='./BUSBRA/BUSBRA/Images/', type=str)
    parser.add_argument('--data_root2', default='./BUSBRA/BUSBRA/Masks/', type=str)
    parser.add_argument('--dataset_name', default='BUSBRA', type=str, choices=['BUSBRA', 'TN3K'])
    
    # Model arguments
    parser.add_argument('--arch', default='vmamba_small', type=str,
                        choices=['vit_tiny','vit_small','vit_base','vit_large',
                                 'swin_tiny','swin_small','swin_base','swin_large',
                                 'resnet50','resnet101','vmamba_small'])
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--window_size', default=7, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0,1,2], type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument("--pretrained_vmamba", default=False, type=utils.bool_flag)
    parser.add_argument('--load_complete_checkpoint', default=False, type=utils.bool_flag, 
                        help='Load complete model checkpoint (backbone+head). If False, loads backbone and head separately.')
    
    # Test arguments
    parser.add_argument('--output_dir', default='.', type=str, help='Directory containing the trained checkpoint')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--ignore_index', default=255, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_predictions', default=False, type=utils.bool_flag, help='Save test predictions to file')
    parser.add_argument('--save_masks', default=True, type=utils.bool_flag, help='Save predicted masks to file')
    parser.add_argument('--cpk_name', default='best', type=str, help='Name of the checkpoint')
    # Distributed arguments
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    
    args = parser.parse_args()
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k,v in sorted(vars(args).items())))
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test_loader, test_ds = load_test_data(args)

    model, embed_dim = load_model(args)

    head = MambaDecoderHead(num_classes=args.num_classes).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    results = {}
    for ck in args.checkpoint_key.split(','):
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = ck.strip()
        
        print(f"\n{'='*60}")
        print(f"Testing checkpoint key: {args_copy.checkpoint_key}")
        print(f"{'='*60}")
        
        if args_copy.load_complete_checkpoint:
            checkpoint = load_complete_checkpoint(model, head, args_copy)
            if checkpoint is None:
                print(f"Skipping checkpoint key: {args_copy.checkpoint_key}")
                continue
        else:
            load_pretrained_weights(model, args_copy)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args_copy.checkpoint_key}_seg_{args.cpk_name}.pth")
            if os.path.isfile(checkpoint_path):
                print(f"Loading head checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                head.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded head checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
            else:
                print(f"No head checkpoint found at {checkpoint_path}")
                continue

        # Freeze all model parameters for testing
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = False
        print("All model parameters (backbone + head) frozen for testing.")

        test_metrics = test_seg(test_loader, model, head, criterion, args_copy, 
                               save_predictions=args.save_predictions)
        
        results[args_copy.checkpoint_key] = test_metrics
        
        test_results = {
            'dataset': args.dataset_name,
            'checkpoint_key': args_copy.checkpoint_key,
            'checkpoint_path': os.path.join(args.output_dir, f"checkpoint_{args_copy.checkpoint_key}_seg.pth"),
            'test_metrics': test_metrics,
            'num_test_samples': len(test_ds),
            'training_info': {
                'best_miou': checkpoint.get('best_miou', 'N/A'),
                'best_dice': checkpoint.get('best_dice', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A')
            }
        }
        
        results_path = os.path.join(args.output_dir, f"test_results_{args_copy.checkpoint_key}_{args.cpk_name}.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {results_path}")
    
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Total test samples: {len(test_ds)}")
    print(f"Architecture: {args.arch}")
    
    for ck, metrics in results.items():
        print(f"\nCheckpoint '{ck}':")
        print(f"  Test mIoU: {metrics['miou']:.4f}")
        print(f"  Test Dice: {metrics['dice']:.4f}")
        print(f"  Test Loss: {metrics['loss']:.4f}")
    
    if len(results) > 1:
        best_ck = max(results.keys(), key=lambda x: results[x]['miou'])
        print(f"\nBest checkpoint by mIoU: '{best_ck}' ({results[best_ck]['miou']:.4f})")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 