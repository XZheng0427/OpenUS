# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import json
import random
import datetime
import subprocess
import argparse
import numpy as np
import torch
import torch.distributed as dist

from collections import defaultdict, deque
from pathlib import Path
from torch import nn
from PIL import ImageFilter, ImageOps, Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as PILImage
import torch.nn.functional as F

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j+self.psz, i+self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new('RGB', (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img

class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size 
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        # img.save('test1.png')
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle((mw * self.psz, 
                            mh * self.psz,
                            (mw + 1) * self.psz,
                            (mh + 1) * self.psz), fill="black")
        # img.save('test2.png')
        return img

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return
    elif pretrained_weights == 'download':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            return
    elif pretrained_weights == 'supervised':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "deit_small_patch16_224-cd65a155.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "deit_base_patch16_224-b5f2ef4d.pth"
        if url is not None:
            print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/" + url)
            msg = model.load_state_dict(state_dict['model'], strict=False)
            print('Supervised weights found at {} and loaded with msg: {}'.format(url, msg))
            return
    print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        print(f"No checkpoint found at {ckp_path}")
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        # args.rank = int(os.environ['SLURM_PROCID'])
        # args.gpu = args.rank % torch.cuda.device_count()
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.getenv('SLURM_NTASKS', 1))
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, 
                return_attn=False, return_rec_loss=False, return_recon_importance=True, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        attn_maps = []  # to store attention maps
        rec_loss = 0
        recon_importance = []
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            # Handle different return types
            if return_attn:
                _out, _attn = self.backbone(inp_x, return_attn=True, **kwargs)
                attn_maps.append(_attn)
            elif return_rec_loss:
                # Handle the case when backbone returns a tuple (output, loss)
                backbone_output = self.backbone(inp_x, return_rec_loss=True, return_recon_importance=return_recon_importance, **kwargs)
                if isinstance(backbone_output, tuple):
                    _out, _rec_loss, _recon_importance = backbone_output
                    rec_loss += _rec_loss
                    if return_recon_importance:
                        recon_importance.append(_recon_importance)   
                else:
                    _out = backbone_output
            else:
                _out = self.backbone(inp_x, **kwargs)
                
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
                
            start_idx = end_idx
            
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        
        # Return appropriate outputs based on parameters
        if return_backbone_feat:
            return output, output_
        elif return_attn:
            return output_, torch.cat(attn_maps) if len(attn_maps) > 1 else attn_maps[0]
        elif return_rec_loss and rec_loss is not None:
            if return_recon_importance:
                return output_, rec_loss, torch.cat(recon_importance)
            else:
                return output_, rec_loss
        else:
            return output_


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def visualize_attention_maps(images, teacher_attn, save_dir="attention_maps", num_samples=5):
    from analyze.utils import visualize, show_mask_on_image
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check dimensions of teacher_attn and determine how to interpret it
    if teacher_attn.dim() == 2:
        # If teacher_attn is 2D, assume it's already in the form [N, HW] where N is number of tokens
        # and reshape appropriately
        H = W = int(np.sqrt(teacher_attn.shape[0]))
        print(f"teacher_attn is 2D with shape {teacher_attn.shape}, interpreting as [N, HW] and reshaping to H={H}, W={W}")
        B = 1  # Assume batch size of 1 in this case
    else:
        # Original assumption: teacher_attn has shape [B, N, HW]
        B, H, W = teacher_attn.shape[0], int(np.sqrt(teacher_attn.shape[-1])), int(np.sqrt(teacher_attn.shape[-1]))
        print(f"B: {B}, H: {H}, W: {W}")
    
    # Process a subset of images in the batch
    for idx in range(min(B, num_samples)):
        if isinstance(images, list):
            # If images is a list, take the idx-th element
            img_item = images[idx] if idx < len(images) else images[0]
            if torch.is_tensor(img_item):
                img = img_item.cpu()
            else:
                # If it's already a PIL image or something else
                img = img_item
        else:
            # Handle tensor cases
            if images.dim() == 4:  # [B, C, H, W]
                img = images[idx].cpu() if idx < images.shape[0] else images[0].cpu()
            elif images.dim() == 5:  # [B, N, C, H, W] where N is number of crops
                img = images[0, idx].cpu() if idx < images.shape[1] else images[0, 0].cpu()  # Take first crop
        
        # Convert to numpy array if it's a tensor
        if torch.is_tensor(img):
            # Denormalize image for visualization
            # Create normalization tensors on the same device as img
            mean = torch.tensor([0.25, 0.25, 0.25], device=img.device).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(-1, 1, 1)
            deimg = img * mean + std
            
            # Check dimension of deimg and handle accordingly
            if deimg.dim() == 3:  # [C, H, W]
                deimg = deimg.permute(1, 2, 0).cpu()  # [C, H, W] -> [H, W, C]
            elif deimg.dim() == 4:  # [N, C, H, W] where N could be number of crops
                deimg = deimg[0].permute(1, 2, 0).cpu()  # Take first crop and convert [C, H, W] -> [H, W, C]
                
            img_array = (deimg * 255).to(torch.uint8).numpy()
        else:
            # If it's already a numpy array or PIL Image
            if isinstance(img, np.ndarray):
                img_array = img
            else:
                img_array = np.array(img)
        
        # Save original image
        os.makedirs(f"{save_dir}/sample_{idx}", exist_ok=True)
        Image.fromarray((deimg * 255).to(torch.uint8).numpy()).save(f"{save_dir}/sample_{idx}/original.jpg")
        
        # Select points to visualize (can be adjusted)
        points = [
            (0.3, 0.3), (0.7, 0.7),
            (0.3, 0.7), (0.7, 0.3),
            (0.1, 0.7), (0.9, 0.3),
            (0.0, 0.0)
        ]
        
        # Visualize attention for each point
        for p_idx, (posx, posy) in enumerate(points):
            # Convert normalized positions to indices
            x_idx = int(posy * H)
            y_idx = int(posx * W)
            pos_idx = x_idx * W + y_idx
            
            # Get attention weights for this position and reshape to spatial dimensions
            if teacher_attn.dim() == 2:
                # If 2D, use different indexing
                attn_map = teacher_attn[pos_idx].view(H, W)
            else:
                # Original 3D indexing
                attn_map = teacher_attn[idx, :, pos_idx].view(H, W)
            
            # Normalize attention weights for better visualization
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # Save attention map
            visualize.visualize_attnmap(
                attn_map, 
                f"{save_dir}/sample_{idx}/attn_pos_{posx:.2f}_{posy:.2f}.jpg",
                colorbar=False, 
                sticks=False
            )
            
            img_tensor = deimg.clone()  # Already a tensor, just make a copy
            
            overlay = show_mask_on_image(
                img_tensor,  # Use tensor version
                attn_map,    # Already a tensor
            )
            
            # Convert overlay (which is now a numpy array) to PIL Image for saving
            Image.fromarray(overlay).save(f"{save_dir}/sample_{idx}/overlay_pos_{posx:.2f}_{posy:.2f}.jpg")
            

def visualize_attention_maps_1(images, teacher_attn, save_dir="attention_maps", num_samples=3, patch_size=4):
    """
    Modified visualization function for VMamba attention maps.
    The output layout is similar to plot_dino_attn_maps: it displays the input image,
    an average (mean) attention map over all heads, and individual head attention maps.
    
    Args:
        images: List of images (PIL Images, numpy arrays, or tensors with shape [C,H,W])
        teacher_attn: Attention maps tensor. Expected shape:
                      - [num_heads, L] (for a single image), or 
                      - [B, num_heads, L] (for a batch), where L = H_feat * W_feat.
        save_dir: Directory to save the plots.
        num_samples: Number of samples (images) to visualize.
        patch_size: Upscaling factor (patch size) to resize attention maps to image resolution.
    """

    

    os.makedirs(save_dir, exist_ok=True)

    # Process teacher_attn: reshape if needed so that we have [B, num_heads, H_feat, W_feat]
    if teacher_attn.dim() == 2:
        # Assume teacher_attn shape is [num_heads, L] for a single image.
        num_heads, L = teacher_attn.shape
        H_feat = W_feat = int(math.sqrt(L))
        teacher_attn = teacher_attn.reshape(1, num_heads, H_feat, W_feat)
    elif teacher_attn.dim() == 3:
        # Assume teacher_attn shape is [B, L] or [B, num_heads, L]?
        # Here we assume the second dimension is the number of heads.
        B, num_heads, L = teacher_attn.shape
        H_feat = W_feat = int(math.sqrt(L))
        teacher_attn = teacher_attn.reshape(B, num_heads, H_feat, W_feat)
    elif teacher_attn.dim() == 4:
        # Already in shape [B, num_heads, H_feat, W_feat]
        B, num_heads, H_feat, W_feat = teacher_attn.shape
    else:
        raise ValueError("Unexpected teacher_attn shape.")
    
    # Process each sample
    for idx in range(num_samples):
        # Get the image (supports list of PIL Images, numpy arrays, or torch tensors)
        if isinstance(images, list):
            img_item = images[idx % len(images)]
        else:
            # assume tensor with shape [B, C, H, W]
            img_item = images[idx % images.shape[0]]
        
        if torch.is_tensor(img_item):
            # Denormalize image for visualization
            # Create normalization tensors on the same device as img_item
            mean = torch.tensor([0.25, 0.25, 0.25], device=img_item.device).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=img_item.device).view(-1, 1, 1)
            deimg = img_item * mean + std
            
            # Check dimension of deimg and handle accordingly
            if deimg.dim() == 3:  # [C, H, W]
                deimg = deimg.permute(1, 2, 0).cpu()  # [C, H, W] -> [H, W, C]
            elif deimg.dim() == 4:  # [N, C, H, W] where N could be number of crops
                deimg = deimg[0].permute(1, 2, 0).cpu()  # Take first crop and convert [C, H, W] -> [H, W, C]
                
            img_np = (deimg * 255).to(torch.uint8).numpy()
        else:
            # If it's already a numpy array or PIL Image
            if isinstance(img_item, np.ndarray):
                img_np = img_item
            else:
                img_np = np.array(img_item)

        # For single image, teacher_attn index 0; for batch use idx.
        if teacher_attn.shape[0] == 1:
            attn = teacher_attn[0]  # shape [num_heads, H_feat, W_feat]
        else:
            attn = teacher_attn[idx]  # shape [num_heads, H_feat, W_feat]
        print(f"attn: {attn.shape}")
        # --- Apply thresholding per head (similar to plot_dino_attn_maps) ---
        # For each head, sort the attention values, compute cumulative sum,
        # and threshold to keep the top mass (here, 60% mass is kept).
        threshold = 0.6
        attn_thresholded = []
        for h in range(num_heads):
            head_attn = attn[h]  # [H_feat, W_feat]
            flat = head_attn.view(-1)
            sorted_vals, sorted_idx = torch.sort(flat)
            sorted_vals = sorted_vals / sorted_vals.sum()
            cum_vals = torch.cumsum(sorted_vals, dim=0)
            # Create a binary mask where the cumulative value exceeds (1 - threshold)
            mask = cum_vals > (1 - threshold)
            # Unsort the mask back to original order
            _, unsort_idx = torch.sort(sorted_idx)
            mask = mask[unsort_idx]
            mask = mask.view(H_feat, W_feat).float()
            attn_thresholded.append(mask)
        attn_thresholded = torch.stack(attn_thresholded, dim=0)  # [num_heads, H_feat, W_feat]

        # --- Interpolate (upsample) attention maps to the original image resolution ---
        # Using the patch_size as scale factor
        attn_up = F.interpolate(attn_thresholded.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        attn_up = attn_up.detach().cpu().numpy()  # shape: [num_heads, H_img, W_img]
        attn_orig = F.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        attn_orig = attn_orig.detach().cpu().numpy()
        attn_mean = np.mean(attn_orig, axis=0)  # average over heads
        print(f"attn_mean: {attn_mean.shape}")
        # --- Plot using matplotlib (similar layout to plot_dino_attn_maps) ---
        fig = plt.figure(figsize=(6, 6), dpi=200)
        # Subplot 1: Original image
        ax = fig.add_subplot(3, 3, 1)
        ax.set_title("Input")
        ax.imshow(img_np)
        ax.axis("off")
        
        # Subplot 2: Mean attention map over heads
        ax = fig.add_subplot(3, 3, 2)
        ax.set_title("Head Mean")
        ax.imshow(attn_mean, cmap="viridis")
        ax.axis("off")
        
        # Subplots for individual head maps (up to 6 heads)
        num_to_plot = min(6, num_heads)
        for i in range(num_to_plot):
            ax = fig.add_subplot(3, 3, i + 4)
            ax.set_title("Head " + str(i + 1))
            ax.imshow(attn_orig[i], cmap="viridis")
            ax.axis("off")
        
        fig.tight_layout()
        out_path = os.path.join(save_dir, f"sample_{idx}_attn_maps.png")
        fig.savefig(out_path)
        plt.close(fig)


def GlobalAttGuidedMask(attention, masking_ratio, b, T, threshold, patch_size):  # attention [B,T,196]
    attention = attention.mean(1)
    N_vis = math.ceil(attention.size(1) * (1 - masking_ratio))

    top_k = int(attention.shape[1] * threshold)
    top_indices = torch.topk(attention, k=top_k, dim=1)[1]
    randomChoice_top_indices = torch.randperm(top_indices.size(1))[:N_vis]
    result_random_top = top_indices[:, randomChoice_top_indices]

    bottom_k = attention.shape[1] - top_k
    bottom_indices = torch.topk(attention, k=bottom_k, largest=False, dim=1)[1]
    randomChoice_bottom_indices = torch.randperm(bottom_indices.size(1))[:0]
    result_random_bottom = bottom_indices[:, randomChoice_bottom_indices]
    vis_idx_ = torch.cat([result_random_top, result_random_bottom], dim=1)

    masks_attn = torch.ones((attention.shape[0], attention.shape[1])).to(attention.device, non_blocking=True)
    masks_attn.scatter_(dim=-1, index=vis_idx_.long(), value=0.0)
    masks_attn = masks_attn.unsqueeze(1).expand(-1, T, -1).reshape(b * T, -1)
    mask = masks_attn.reshape(-1, 224 // patch_size, 224 // patch_size)  # [[BT,14,14],[BT,14,14]]
    return vis_idx_, mask

   
def GlobalAttGuidedMask_2(attention, masking_ratio, b, T, threshold, patch_size):  # attention [B,T,196]
    attention = attention.mean(1)
    total_tokens = attention.size(1)
    N_total_mask = math.ceil(total_tokens * masking_ratio)
    
    # Get top_k tokens with highest attention
    top_k = int(total_tokens * threshold)
    top_indices = torch.topk(attention, k=top_k, dim=1)[1]
    
    # If we need more masks beyond the top_k tokens
    remaining_masks_needed = N_total_mask - top_k
    
    if remaining_masks_needed > 0:
        # For each batch item, get non-top indices
        additional_mask_indices = []
        
        for i in range(attention.shape[0]):
            # Create a mask of all tokens
            is_top = torch.zeros(total_tokens, device=attention.device, dtype=torch.bool)
            # Mark top tokens as True
            is_top[top_indices[i]] = True
            # Get indices of non-top tokens
            non_top_idx = torch.nonzero(~is_top, as_tuple=True)[0]
            # previous random sampling
            perm = torch.randperm(len(non_top_idx), device=attention.device)[:remaining_masks_needed]
            rand_non_top = non_top_idx[perm]
            
            additional_mask_indices.append(rand_non_top)
        
        additional_mask_indices = torch.stack(additional_mask_indices)

        mask_indices = torch.cat([top_indices, additional_mask_indices], dim=1)
    else:
        mask_indices = top_indices[:, :N_total_mask]

    masks_attn = torch.zeros((attention.shape[0], attention.shape[1])).to(attention.device, non_blocking=True)
    masks_attn.scatter_(dim=-1, index=mask_indices.long(), value=1.0)
    
    masks_attn = masks_attn.unsqueeze(1).expand(-1, T, -1).reshape(b * T, -1)
    mask = masks_attn.reshape(-1, 224 // patch_size, 224 // patch_size)  # [[BT,14,14],[BT,14,14]]
    
    return mask_indices, mask

def GlobalAttGuidedMask_3vis(attention, masking_ratio, b, T, threshold, patch_size):  # attention [B,T,196]
    attention = attention.mean(1)
    total_tokens = attention.size(1)
    N_total_mask = math.ceil(total_tokens * masking_ratio)
    
    # Get top_k tokens with highest attention
    top_k = int(N_total_mask * threshold)
    top_indices = torch.topk(attention, k=top_k, dim=1)[1]
    
    mask_indices = top_indices
    
    # Create the mask tensor (1 = masked, 0 = visible)
    masks_attn = torch.zeros((attention.shape[0], attention.shape[1])).to(attention.device, non_blocking=True)
    masks_attn.scatter_(dim=-1, index=mask_indices.long(), value=1.0)
    
    # Expand and reshape
    masks_attn = masks_attn.unsqueeze(1).expand(-1, T, -1).reshape(b * T, -1)
    mask = masks_attn.reshape(-1, 224 // patch_size, 224 // patch_size)  # [[BT,14,14],[BT,14,14]]
    
    return mask_indices, mask


def visualize_attention_maps_2(images, teacher_attn, save_dir="attention_maps", 
                               num_samples=3, patch_size=4, masking_ratio=0.65, mask_threshold=0.001, epoch=0, total_epochs=150, image_names=None):

    os.makedirs(save_dir, exist_ok=True)

    if teacher_attn.dim() == 2:
        num_heads, L = teacher_attn.shape
        H_feat = W_feat = int(math.sqrt(L))
        teacher_attn = teacher_attn.reshape(1, num_heads, H_feat, W_feat)
    elif teacher_attn.dim() == 3:
        B, num_heads, L = teacher_attn.shape
        H_feat = W_feat = int(math.sqrt(L))
        teacher_attn = teacher_attn.reshape(B, num_heads, H_feat, W_feat)
    elif teacher_attn.dim() == 4:
        B, num_heads, H_feat, W_feat = teacher_attn.shape
    else:
        raise ValueError("Unexpected teacher_attn shape.")
    print(f"teacher_attn: {teacher_attn.shape}")
    for idx in range(num_samples):
        if isinstance(images, list):
            img_item = images[idx % len(images)]
        else:
            img_item = images[idx % images.shape[0]]
        
        if torch.is_tensor(img_item):
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_item.device).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_item.device).view(-1, 1, 1)
            deimg = img_item * std + mean
            
            if deimg.dim() == 3:  # [C, H, W]
                deimg = deimg.permute(1, 2, 0).cpu()
            elif deimg.dim() == 4:  # [N, C, H, W]
                deimg = deimg[0].permute(1, 2, 0).cpu()
            
            deimg = torch.clamp(deimg, 0, 1)
            img_np = (deimg * 255).to(torch.uint8).numpy()
        else:
            if isinstance(img_item, np.ndarray):
                img_np = img_item
            else:
                img_np = np.array(img_item)

        if teacher_attn.shape[0] == 1:
            attn = teacher_attn[0]  # shape [num_heads, H_feat, W_feat]
        else:
            attn = teacher_attn[idx]  # shape [num_heads, H_feat, W_feat]
        print(f"attn: {attn.shape}")
        thresh = 0.6
        attn_thresholded = []
        for h in range(num_heads):
            head_attn = attn[h]  # [H_feat, W_feat]
            flat = head_attn.view(-1)
            sorted_vals, sorted_idx = torch.sort(flat)
            sorted_vals = sorted_vals / sorted_vals.sum()
            cum_vals = torch.cumsum(sorted_vals, dim=0)
            mask_local = cum_vals > (1 - thresh)
            _, unsort_idx = torch.sort(sorted_idx)
            mask_local = mask_local[unsort_idx]
            mask_local = mask_local.view(H_feat, W_feat).float()
            attn_thresholded.append(mask_local)
        attn_thresholded = torch.stack(attn_thresholded, dim=0)  # [num_heads, H_feat, W_feat]

        attn_up = F.interpolate(attn_thresholded.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        attn_up = attn_up.detach().cpu().numpy()  # shape: [num_heads, H_img, W_img]
        attn_orig = F.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        attn_orig = attn_orig.detach().cpu().numpy()
        attn_mean = np.mean(attn_orig, axis=0)  # average over heads

        global_att = attn.mean(dim=0)  
        global_att = global_att.view(1, 1, -1)
        m_t = 0.1 + (epoch / total_epochs) * (0.9 - 0.1)
        vis_idx, mask_generated = GlobalAttGuidedMask_2(global_att, masking_ratio, 1, 1, m_t, patch_size*8)
        vis_idx_attn, mask_generated_attn = GlobalAttGuidedMask_3vis(global_att, masking_ratio, 1, 1, m_t, patch_size*8)
        print(f"mask_generated: {mask_generated.shape}")
        mask_upsampled = F.interpolate(mask_generated.unsqueeze(0).float(), size=(img_np.shape[0], img_np.shape[1]), mode="nearest")[0,0].cpu().numpy()
        mask_upsampled_attn = F.interpolate(mask_generated_attn.unsqueeze(0).float(), size=(img_np.shape[0], img_np.shape[1]), mode="nearest")[0,0].cpu().numpy()
        masked_img = img_np.copy()
        
        overlay_color = np.array([50, 50, 50])*0.5  # Dark Gray
        overlay_color_red = np.array([255, 0, 0])*0.8  # Light Red
        mask_overlay = np.zeros_like(masked_img)
        mask_overlay[mask_upsampled > 0.5] = overlay_color
        
        alpha = 0.4
        masked_regions = mask_upsampled > 0.5
        masked_img[masked_regions] = (1 - alpha) * masked_img[masked_regions] + alpha * mask_overlay[masked_regions]
        
        mask_overlay_attn = np.zeros_like(masked_img)
        mask_overlay_attn[mask_upsampled_attn > 0.5] = overlay_color_red
        alpha = 0.4
        masked_regions_attn = mask_upsampled_attn > 0.5
        masked_img[masked_regions_attn] = (1 - alpha) * masked_img[masked_regions_attn] + alpha * mask_overlay_attn[masked_regions_attn]
        
        alpha_unmasked = 0.3
        unmasked_regions = ~masked_regions
        masked_img[unmasked_regions] = (1 - alpha_unmasked) * masked_img[unmasked_regions] + alpha_unmasked * np.array([255, 255, 255])
        
        attn_overlay_img = img_np.copy().astype(np.float32)
        
        attn_mean_normalized = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)
        
        attention_colored = cm.jet(attn_mean_normalized)[:, :, :3]
        attention_colored = (attention_colored * 255).astype(np.uint8)
        
        attention_colored_pil = PILImage.fromarray(attention_colored)
        attention_colored_upsampled = attention_colored_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
        attention_colored_upsampled = np.array(attention_colored_upsampled)
        
        alpha_attention = 0.4
        attn_overlay_img = (1 - alpha_attention) * attn_overlay_img + alpha_attention * attention_colored_upsampled.astype(np.float32)
        attn_overlay_img = np.clip(attn_overlay_img, 0, 255).astype(np.uint8)
        
        attn_masked_img = img_np.copy()
        
        mask_overlay_attn_only = np.zeros_like(attn_masked_img)
        mask_overlay_attn_only[mask_upsampled_attn > 0.5] = overlay_color  # Dark grey
        alpha = 0.4
        masked_regions_attn_only = mask_upsampled_attn > 0.5
        attn_masked_img[masked_regions_attn_only] = (1 - alpha) * attn_masked_img[masked_regions_attn_only] + alpha * mask_overlay_attn_only[masked_regions_attn_only]
        
        alpha_unmasked = 0.3
        unmasked_regions_attn_only = ~masked_regions_attn_only
        attn_masked_img[unmasked_regions_attn_only] = (1 - alpha_unmasked) * attn_masked_img[unmasked_regions_attn_only] + alpha_unmasked * np.array([255, 255, 255])

        fig = plt.figure(figsize=(20, 9), dpi=200)
        
        ax = fig.add_subplot(3, 5, 1)
        ax.set_title("Input")
        ax.imshow(img_np)
        ax.axis("off")
        
        ax = fig.add_subplot(3, 5, 2)
        ax.set_title("Head Mean")
        ax.imshow(attn_mean, cmap="viridis")
        ax.axis("off")
        
        ax = fig.add_subplot(3, 5, 3)
        ax.set_title("Attention Overlay")
        ax.imshow(attn_overlay_img)
        ax.axis("off")
        
        ax = fig.add_subplot(3, 5, 4)
        ax.set_title("Global Mask")
        ax.imshow(mask_generated[0].cpu(), cmap="gray")
        ax.axis("off")
        
        ax = fig.add_subplot(3, 5, 5)
        ax.set_title("Masked Image")
        ax.imshow(masked_img)
        ax.axis("off")
        
        ax = fig.add_subplot(3, 5, 6)
        ax.set_title("Attention Mask Only")
        ax.imshow(attn_masked_img)
        ax.axis("off")
        
        num_to_plot = min(9, num_heads)
        for i in range(num_to_plot):
            ax = fig.add_subplot(3, 5, 7 + i)
            ax.set_title("Head " + str(i + 1))
            ax.imshow(attn_orig[i], cmap="viridis")
            ax.axis("off")
        
        fig.tight_layout()
        
        if image_names is not None and idx < len(image_names):
            base_name = os.path.splitext(image_names[idx])[0]
            out_path = os.path.join(save_dir, f"{base_name}_attn_maps.png")
        else:
            out_path = os.path.join(save_dir, f"sample_{idx}_attn_maps.png")
        
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved attention visualization to: {out_path}")