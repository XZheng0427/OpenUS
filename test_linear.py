# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test linear classifier and calculate detailed performance metrics.
Based on eval_linear.py but focused on testing only.
"""

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import utils
import models
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms
from dataset.dataset_fetal_planes_pytorch import FetalPlanesDataset
from dataset.dataset_busi import BUSIDataset
from vmamba_models.dino_vmamba import dinov2_vmamba_small, Backbone_VSSM, Backbone_DINOv2_VSSM, Backbone_DINOv2_VSSM_2

def test_linear(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    test_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Load test dataset
    if args.dataset == 'fetal_planes':
        dataset_test = FetalPlanesDataset(data_root=args.data_path, split='test', transform=test_transform)
    elif args.dataset == 'busi':
        dataset_test = BUSIDataset(data_root=args.data_path, split='test', transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    print(f"Test data loaded with {len(dataset_test)} images.")

    if 'swin' in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](
            window_size=args.window_size,
            patch_size=args.patch_size,
            num_classes=0)
        embed_dim = model.num_features
    elif args.arch == 'vmamba_small':
        if args.pretrained_vmamba:
            model = Backbone_DINOv2_VSSM_2(pretrained='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth')
            embed_dim = model.dims[-1]
        else:
            model = dinov2_vmamba_small(
                    patch_size=args.patch_size, 
                    return_all_tokens=True,
                    masked_im_modeling=False
                    )
            embed_dim = model.dims[-1]
    else:
        model = models.__dict__[args.arch](
            patch_size=args.patch_size, 
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens==1)
        embed_dim = model.embed_dim
    
    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
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
                    print("WARNING: Could not load pretrained weights. Using random init.")
            else:
                print(f"No key '{args.checkpoint_key}' found. Available keys:", list(checkpoint.keys()))
                print("WARNING: Could not load pretrained weights. Using random init.")
        else:
            print(f"No pretrained weights found at {args.pretrained_weights}")
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    
    if 'swin' in args.arch:
        num_features = []
        for i, d in enumerate(model.depths):
            num_features += [int(model.embed_dim * 2 ** i)] * d
        feat_dim = sum(num_features[-args.n_last_blocks:])
    elif args.arch == 'vmamba_small':
        feat_dim = embed_dim
    else:
        feat_dim = embed_dim * (args.n_last_blocks * int(args.avgpool_patchtokens != 1) + \
            int(args.avgpool_patchtokens > 0))
        
    linear_classifier = LinearClassifier(feat_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        print(f"Loading trained linear classifier from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Best training accuracy: {checkpoint.get('best_acc', 'unknown')}")
        else:
            state_dict = checkpoint
        
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("Removing 'module.' prefix from checkpoint keys...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        linear_classifier.load_state_dict(state_dict)
        print("Successfully loaded trained linear classifier")
    else:
        raise ValueError(f"No checkpoint found at {args.checkpoint_path}")
    
    linear_classifier.eval()
    
    print("Starting evaluation...")
    test_stats = evaluate_with_metrics(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.num_labels, args)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {test_stats['accuracy']:.4f}")
    print(f"Macro-averaged Precision: {test_stats['precision_macro']:.4f}")
    print(f"Macro-averaged Recall: {test_stats['recall_macro']:.4f}")
    print(f"Macro-averaged F1: {test_stats['f1_macro']:.4f}")
    print(f"Weighted-averaged Precision: {test_stats['precision_weighted']:.4f}")
    print(f"Weighted-averaged Recall: {test_stats['recall_weighted']:.4f}")
    print(f"Weighted-averaged F1: {test_stats['f1_weighted']:.4f}")
    
    print("\nPer-class Results:")
    for i, (prec, rec, f1) in enumerate(zip(test_stats['precision_per_class'], 
                                          test_stats['recall_per_class'], 
                                          test_stats['f1_per_class'])):
        print(f"Class {i}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(test_stats['classification_report'])
    
    print("\nConfusion Matrix:")
    print(test_stats['confusion_matrix'])
    
    if args.output_dir:
        results_file = os.path.join(args.output_dir, f"test_results_{args.checkpoint_key}.txt")
        with open(results_file, 'w') as f:
            f.write("TEST RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Accuracy: {test_stats['accuracy']:.4f}\n")
            f.write(f"Macro-averaged Precision: {test_stats['precision_macro']:.4f}\n")
            f.write(f"Macro-averaged Recall: {test_stats['recall_macro']:.4f}\n")
            f.write(f"Macro-averaged F1: {test_stats['f1_macro']:.4f}\n")
            f.write(f"Weighted-averaged Precision: {test_stats['precision_weighted']:.4f}\n")
            f.write(f"Weighted-averaged Recall: {test_stats['recall_weighted']:.4f}\n")
            f.write(f"Weighted-averaged F1: {test_stats['f1_weighted']:.4f}\n")
            f.write("\nPer-class Results:\n")
            for i, (prec, rec, f1) in enumerate(zip(test_stats['precision_per_class'], 
                                                  test_stats['recall_per_class'], 
                                                  test_stats['f1_per_class'])):
                f.write(f"Class {i}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n")
            f.write("\nDetailed Classification Report:\n")
            f.write(test_stats['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(test_stats['confusion_matrix']))
        print(f"\nResults saved to {results_file}")


@torch.no_grad()
def evaluate_with_metrics(test_loader, model, linear_classifier, n, avgpool, num_labels, args):
    linear_classifier.eval()
    
    all_predictions = []
    all_targets = []
    
    for inp, target in test_loader:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                if avgpool == 0:
                    # norm(x[:, 0])
                    output = [x[:, 0] for x in intermediate_output]
                elif avgpool == 1:
                    # x[:, 1:].mean(1)
                    output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
                elif avgpool == 2:
                    # norm(x[:, 0]) + x[:, 1:].mean(1)
                    output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
                else:
                    assert False, "Unknown avgpool type {}".format(avgpool)              
            elif args.arch == 'vmamba_small':
                output = model(inp)
                if avgpool == 0:
                    output = [output[:, 0]]
                elif avgpool == 1:
                    output = output[:, 1:]
                    output = [torch.mean(output, dim=1)]
                elif avgpool == 2:
                    output = [output[:, 0] + torch.mean(output[:, 1:], dim=1)]
                else:
                    assert False, "Unknown avgpool type {}".format(avgpool)
                    
            output = torch.cat(output, dim=-1)
        
        logits = linear_classifier(output)
        predictions = torch.argmax(logits, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0)
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0)
    
    class_report = classification_report(all_targets, all_predictions, zero_division=0)
    
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'targets': all_targets
    }


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test linear classifier with comprehensive metrics')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base/large-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101', 'dalle_encoder', 'vmamba_small'], help='Architecture.')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to the trained linear classifier checkpoint')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument('--data_path', default='.', type=str,
        help='Please specify path to the dataset.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save results')
    parser.add_argument('--num_labels', default=4, type=int, help='Number of labels for linear classifier')
    parser.add_argument("--pretrained_vmamba", default=False, type=utils.bool_flag)
    parser.add_argument('--dataset', default='fetal_planes', type=str, choices=['busi', 'fetal_planes'], help='Dataset to use')
    
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    test_linear(args) 