import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class SwappingScan_multiview(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, x2: torch.Tensor):
        # B, C, H, W -> B, 2, C, H*W
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        
        x = x.view(B, C, -1)  # shape [B, C, H*W]
        x2 = x2.view(B, C, -1)  # shape [B, C, H*W]
        exchange_mask = torch.arange(C) % 2 == 0   # shape [C]
        exchange_mask = exchange_mask.unsqueeze(0).expand(B, -1)  # shape [B, C]
        
        out_x = torch.zeros_like(x)  # shape [B, C, N]
        out_x2 = torch.zeros_like(x2)  # shape [B, C, N]

        out_x[~exchange_mask, ...] = x[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
       
        out_x[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x[exchange_mask, ...]
        
        xs_fuse = x.new_empty((B, 2, C, H * W))
        xs_fuse[:, 0] = out_x
        xs_fuse[:, 1] = out_x2
        
        return xs_fuse

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, 2, d, h, w)
        B, C, H, W = ctx.shape
        
        return ys[:, 0].view(B, -1, H, W), ys[:, 1].view(B, -1, H, W)


class SwappingMerge_multiview(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, L = ys.shape
        # ctx.shape = (H, W)
        out_x  = ys[:, 0].contiguous()  # shape [B, C, L]
        out_x2 = ys[:, 1].contiguous()  # shape [B, C, L]
        return out_x, out_x2
    
    @staticmethod
    def backward(ctx, x: torch.Tensor, x2: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x2
        return xs, None, None


class ConcatScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, x2: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x.flatten(2, 3), x2.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H*W].view(B, -1, H, W), ys[:, :, H*W:2*H*W].view(B, -1, H, W) 
    
         
class ConcatMerge_multimodal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, L = ys.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
        # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
        return ys[:, :, 0:L//2], ys[:, :, L//2:L]
    
    @staticmethod
    def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x1.shape
        xs = x1.new_empty((B, 2, C, 2*L))
        xs[:, 0] = torch.cat([x1, x2], dim=2)
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 2, C, 2*L)
        return xs, None, None


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class Cross_SS2Dv2(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_sec = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y, H=H, W=W,
            ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x, x2: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
        z = self.act(z)
        x2 = self.in_proj_sec(x2)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
        x = self.act(x)
        x2 = self.act(x2)
        y = self.forward_corev2(x)
        y2 = self.forward_corev2(x2)
        
        y = y * z
        y2 = y2 * z
        y_fusion = y + y2
        out = self.dropout(self.out_proj(y_fusion))
        return out


class FusionBlock_v2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        norm_layer: Callable[..., nn.Module],
        attn_drop_rate: float,
        d_state: int,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = Cross_SS2Dv2(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x = self.drop_path(self.self_attention(x1_perm, x2_perm))
        x = x.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x_out = x1 + x
        return x_out

class CSSFVSSLayer_v2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 1,
        drop_path = 0.,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        attn_drop_rate: float = 0.,
        d_state: int = 16,
        downsampling=None,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock_v2(
                hidden_dim=hidden_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                **kwargs,
            )
            for i in range(depth)
        ])

        self.downsampling = downsampling if downsampling != 1 else None

    def downsample(self, z):
        if self.downsampling is not None:
            # downsample
            if z.ndim == 3:
                pool_fn = nn.functional.max_pool1d
            elif z.ndim == 4:
                pool_fn = nn.functional.max_pool2d
            elif z.ndim == 5:
                pool_fn = nn.functional.max_pool3d
            else:
                raise ValueError('Downsampling expects 1D, 2D or 3D input.')
            z = pool_fn(z, self.downsampling, ceil_mode=True)
        return z

    def forward(self, x1, x2):
        for blk in self.blocks:
            x1 = blk(x1, x2)
        return x1


class Cross_SS2Dv4(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        # tags for forward_type ==============================
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        # FORWARD_TYPES = dict(
        #     v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
        #     v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
        #     v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
        #     v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"
        #     v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"
        #     # ===============================
        #     v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
        #     v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
        #     v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
        #     v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
        #     # ===============================
        #     v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
        #     v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        # )
        # self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        # d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        # self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_sec = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            def scan_rowcol(
                x: torch.Tensor, 
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # (2*c)
                _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width = True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight = self.x_proj_weight.view(4, -1, D)[:2].contiguous(), 
                proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3) # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight = self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
                proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x, x2: torch.Tensor, **kwargs):
        x_fuse = (x + x2)/2
        x = self.in_proj_sec(x)
        x_fuse = self.in_proj_sec(x_fuse)
       
        x2 = self.in_proj_sec(x2)

        z = self.act(x_fuse)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()

        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
            x_fuse = self.conv2d(x_fuse)

        x = self.act(x)
        x2 = self.act(x2)
        x_fuse = self.act(x_fuse)

        y = self.forward_corev2(x)
        y2 = self.forward_corev2(x2)
        y_fuse = self.forward_corev2(x_fuse)
        
        y = y * z
        y2 = y2 * z
        y_fuse = y_fuse * z
        y_fusion3 = y + y2 + y_fuse
        out = self.dropout(self.out_proj(y_fusion3))
        return out


class FusionBlock_v4(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        norm_layer: Callable[..., nn.Module],
        attn_drop_rate: float,
        d_state: int,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = Cross_SS2Dv4(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x = self.drop_path(self.self_attention(x1_perm, x2_perm))
        x = x.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x_out = x1 + x2 + x
        return x_out


class CSSFVSSLayer_v4(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 1,
        drop_path = 0.,
        norm_layer: Callable[..., nn.Module] = LayerNorm2d, #nn.BatchNorm2d,
        attn_drop_rate: float = 0.,
        d_state: int = 16,
        downsampling=None,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock_v4(
                hidden_dim=hidden_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                **kwargs,
            )
            for i in range(depth)
        ])

        self.downsampling = downsampling if downsampling != 1 else None

    def downsample(self, z):
        if self.downsampling is not None:
            # downsample
            if z.ndim == 3:
                pool_fn = nn.functional.max_pool1d
            elif z.ndim == 4:
                pool_fn = nn.functional.max_pool2d
            elif z.ndim == 5:
                pool_fn = nn.functional.max_pool3d
            else:
                raise ValueError('Downsampling expects 1D, 2D or 3D input.')
            z = pool_fn(z, self.downsampling, ceil_mode=True)
        return z

    def forward(self, x1, x2):
        for blk in self.blocks:
            x1 = blk(x1, x2)
        return x1



class Cross_SS2Dv5(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_sec = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        x_fuse: torch.Tensor=None,  
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        # ==============================
        xs_fuse = cross_scan_fn(x_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_fuse_dbl = F.conv1d(xs_fuse.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = F.conv1d(dts_fuse.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_fuse_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_fuse, self.x_proj_weight)
            if x_proj_bias is not None:
                x_fuse_dbl = x_fuse_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = torch.einsum("b k r l, k d r -> b k d l", dts_fuse, self.dt_projs_weight)

        xs_fuse = xs_fuse.view(B, -1, L)
        dts_fuse = dts_fuse.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs_fuse = Bs_fuse.contiguous().view(B, K, N, L)
        Cs_fuse = Cs_fuse.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        ys_fuse: torch.Tensor = selective_scan(
            xs_fuse, dts_fuse, As, Bs_fuse, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_fuse: torch.Tensor = cross_merge_fn(ys_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_fuse = y_fuse.view(B, -1, H, W)
        if not channel_first:
            y_fuse = y_fuse.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_fuse = out_norm(y_fuse)
        
        # ==============================
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        # ==============================
        xs_2 = cross_scan_fn(x2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_2_dbl = F.conv1d(xs_2.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = F.conv1d(dts_2.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_2_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_2, self.x_proj_weight)
            if x_proj_bias is not None:
                x_2_dbl = x_2_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = torch.einsum("b k r l, k d r -> b k d l", dts_2, self.dt_projs_weight)

        xs_2 = xs_2.view(B, -1, L)
        dts_2 = dts_2.contiguous().view(B, -1, L)
        Bs_2 = Bs_2.contiguous().view(B, K, N, L)
        Cs_2 = Cs_2.contiguous().view(B, K, N, L)

        ys_2: torch.Tensor = selective_scan(
            xs_2, dts_2, As, Bs_2, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_2: torch.Tensor = cross_merge_fn(ys_2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_2 = y_2.view(B, -1, H, W)
        if not channel_first:
            y_2 = y_2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_2 = out_norm(y_2)
        
        return y.to(x.dtype), y_2.to(x2.dtype), y_fuse.to(x_fuse.dtype)

    def forward(self, x, x2: torch.Tensor, **kwargs):
        x_fuse = (x + x2)/2
        
        x = self.in_proj_sec(x)
        x2 = self.in_proj_sec(x2)
        x_fuse = self.in_proj_sec(x_fuse)

        z = self.act(x_fuse)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()

        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
            x_fuse = self.conv2d(x_fuse)

        x = self.act(x)
        x2 = self.act(x2)
        x_fuse = self.act(x_fuse)

        y, y2, y_fuse = self.forward_corev2(x, x2, x_fuse)
        
        y = y * z
        y2 = y2 * z
        y_fuse = y_fuse * z
        y_fusion3 = y + y2 + y_fuse
        out = self.dropout(self.out_proj(y_fusion3))
        return out


class FusionBlock_v5(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        norm_layer: Callable[..., nn.Module],
        attn_drop_rate: float,
        d_state: int,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = Cross_SS2Dv5(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x = self.drop_path(self.self_attention(x1_perm, x2_perm))
        x = x.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x_out = x1 + x2 + x
        return x_out


class CSSFVSSLayer_v5(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 1,
        drop_path = 0.,
        norm_layer: Callable[..., nn.Module] = LayerNorm2d, #nn.BatchNorm2d,
        attn_drop_rate: float = 0.,
        d_state: int = 16,
        downsampling=None,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock_v5(
                hidden_dim=hidden_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                **kwargs,
            )
            for i in range(depth)
        ])

        self.downsampling = downsampling if downsampling != 1 else None

    def downsample(self, z):
        if self.downsampling is not None:
            # downsample
            if z.ndim == 3:
                pool_fn = nn.functional.max_pool1d
            elif z.ndim == 4:
                pool_fn = nn.functional.max_pool2d
            elif z.ndim == 5:
                pool_fn = nn.functional.max_pool3d
            else:
                raise ValueError('Downsampling expects 1D, 2D or 3D input.')
            z = pool_fn(z, self.downsampling, ceil_mode=True)
        return z

    def forward(self, x1, x2):
        for blk in self.blocks:
            x1 = blk(x1, x2)
        return x1


class Cross_SS2Dv6(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        self.in_proj = nn.Linear(self.d_model*2, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_sec = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        x_fuse: torch.Tensor=None,  
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        # ==============================
        xs_fuse = cross_scan_fn(x_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_fuse_dbl = F.conv1d(xs_fuse.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = F.conv1d(dts_fuse.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_fuse_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_fuse, self.x_proj_weight)
            if x_proj_bias is not None:
                x_fuse_dbl = x_fuse_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = torch.einsum("b k r l, k d r -> b k d l", dts_fuse, self.dt_projs_weight)

        xs_fuse = xs_fuse.view(B, -1, L)
        dts_fuse = dts_fuse.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs_fuse = Bs_fuse.contiguous().view(B, K, N, L)
        Cs_fuse = Cs_fuse.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        ys_fuse: torch.Tensor = selective_scan(
            xs_fuse, dts_fuse, As, Bs_fuse, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_fuse: torch.Tensor = cross_merge_fn(ys_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_fuse = y_fuse.view(B, -1, H, W)
        if not channel_first:
            y_fuse = y_fuse.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_fuse = out_norm(y_fuse)
        
        # ==============================
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        # ==============================
        xs_2 = cross_scan_fn(x2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_2_dbl = F.conv1d(xs_2.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = F.conv1d(dts_2.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_2_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_2, self.x_proj_weight)
            if x_proj_bias is not None:
                x_2_dbl = x_2_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = torch.einsum("b k r l, k d r -> b k d l", dts_2, self.dt_projs_weight)

        xs_2 = xs_2.view(B, -1, L)
        dts_2 = dts_2.contiguous().view(B, -1, L)
        Bs_2 = Bs_2.contiguous().view(B, K, N, L)
        Cs_2 = Cs_2.contiguous().view(B, K, N, L)

        ys_2: torch.Tensor = selective_scan(
            xs_2, dts_2, As, Bs_2, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_2: torch.Tensor = cross_merge_fn(ys_2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_2 = y_2.view(B, -1, H, W)
        if not channel_first:
            y_2 = y_2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_2 = out_norm(y_2)
        
        return y.to(x.dtype), y_2.to(x2.dtype), y_fuse.to(x_fuse.dtype)

    def forward(self, x, x2: torch.Tensor, **kwargs):
        x_fuse = torch.cat([x, x2], dim=-1)
        x = self.in_proj_sec(x)
        x2 = self.in_proj_sec(x2)
        x_fuse = self.in_proj(x_fuse)

        z = self.act(x_fuse)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()

        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
            x_fuse = self.conv2d(x_fuse)

        x = self.act(x)
        x2 = self.act(x2)
        x_fuse = self.act(x_fuse)

        y, y2, y_fuse = self.forward_corev2(x, x2, x_fuse)
        
        y = y * z
        y2 = y2 * z
        y_fuse = y_fuse * z
        y_fusion3 = y + y2 + y_fuse
        out = self.dropout(self.out_proj(y_fusion3))
        return out


class FusionBlock_v6(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        norm_layer: Callable[..., nn.Module],
        attn_drop_rate: float,
        d_state: int,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = Cross_SS2Dv6(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x = self.drop_path(self.self_attention(x1_perm, x2_perm))
        x = x.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x_out = x1 + x2 + x
        return x_out


class CSSFVSSLayer_v6(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 1,
        drop_path = 0.,
        norm_layer: Callable[..., nn.Module] = LayerNorm2d, #nn.BatchNorm2d,
        attn_drop_rate: float = 0.,
        d_state: int = 16,
        downsampling=None,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock_v6(
                hidden_dim=hidden_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                **kwargs,
            )
            for i in range(depth)
        ])

        self.downsampling = downsampling if downsampling != 1 else None

    def downsample(self, z):
        if self.downsampling is not None:
            # downsample
            if z.ndim == 3:
                pool_fn = nn.functional.max_pool1d
            elif z.ndim == 4:
                pool_fn = nn.functional.max_pool2d
            elif z.ndim == 5:
                pool_fn = nn.functional.max_pool3d
            else:
                raise ValueError('Downsampling expects 1D, 2D or 3D input.')
            z = pool_fn(z, self.downsampling, ceil_mode=True)
        return z

    def forward(self, x1, x2):
        for blk in self.blocks:
            x1 = blk(x1, x2)
        return x1
    
    
class Cross_SS2Dv7(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_sec = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        x_fuse: torch.Tensor=None,  
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        # ==============================
        xs_fuse = cross_scan_fn(x_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_fuse_dbl = F.conv1d(xs_fuse.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = F.conv1d(dts_fuse.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_fuse_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_fuse, self.x_proj_weight)
            if x_proj_bias is not None:
                x_fuse_dbl = x_fuse_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = torch.einsum("b k r l, k d r -> b k d l", dts_fuse, self.dt_projs_weight)

        xs_fuse = xs_fuse.view(B, -1, L)
        dts_fuse = dts_fuse.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs_fuse = Bs_fuse.contiguous().view(B, K, N, L)
        Cs_fuse = Cs_fuse.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        ys_fuse: torch.Tensor = selective_scan(
            xs_fuse, dts_fuse, As, Bs_fuse, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_fuse: torch.Tensor = cross_merge_fn(ys_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_fuse = y_fuse.view(B, -1, H, W)
        if not channel_first:
            y_fuse = y_fuse.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_fuse = out_norm(y_fuse)
        
        # ==============================
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        # ==============================
        xs_2 = cross_scan_fn(x2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_2_dbl = F.conv1d(xs_2.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = F.conv1d(dts_2.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_2_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_2, self.x_proj_weight)
            if x_proj_bias is not None:
                x_2_dbl = x_2_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = torch.einsum("b k r l, k d r -> b k d l", dts_2, self.dt_projs_weight)

        xs_2 = xs_2.view(B, -1, L)
        dts_2 = dts_2.contiguous().view(B, -1, L)
        Bs_2 = Bs_2.contiguous().view(B, K, N, L)
        Cs_2 = Cs_2.contiguous().view(B, K, N, L)

        ys_2: torch.Tensor = selective_scan(
            xs_2, dts_2, As, Bs_2, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_2: torch.Tensor = cross_merge_fn(ys_2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_2 = y_2.view(B, -1, H, W)
        if not channel_first:
            y_2 = y_2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_2 = out_norm(y_2)
        
        return y.to(x.dtype), y_2.to(x2.dtype), y_fuse.to(x_fuse.dtype)

    def forward(self, x, x2: torch.Tensor, **kwargs):
        x_fuse = (x + x2)/2
        
        x = self.in_proj_sec(x)
        x2 = self.in_proj_sec(x2)
        x_fuse = self.in_proj_sec(x_fuse)

        z = self.act(x_fuse)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()

        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
            x_fuse = self.conv2d(x_fuse)

        x = self.act(x)
        x2 = self.act(x2)
        x_fuse = self.act(x_fuse)

        y, y2, y_fuse = self.forward_corev2(x, x2, x_fuse)
        
        y = y * z
        y2 = y2 * z
        out = self.dropout(self.out_proj(y))
        out2 = self.dropout(self.out_proj(y2))
        
        return out, out2


class FusionBlock_v7(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        norm_layer: Callable[..., nn.Module],
        attn_drop_rate: float,
        d_state: int,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = Cross_SS2Dv7(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x1_atten, x2_atten = self.drop_path(self.self_attention(x1_perm, x2_perm))
        x1_atten = x1_atten.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x2_atten = x2_atten.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x1_out = x1 + x1_atten
        x2_out = x2 + x2_atten
        return x1_out, x2_out


class CSSFVSSLayer_v7(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 1,
        drop_path = 0.,
        norm_layer: Callable[..., nn.Module] = LayerNorm2d, #nn.BatchNorm2d,
        attn_drop_rate: float = 0.,
        d_state: int = 16,
        downsampling=None,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock_v7(
                hidden_dim=hidden_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                **kwargs,
            )
            for i in range(depth)
        ])

        self.downsampling = downsampling if downsampling != 1 else None

    def downsample(self, z):
        if self.downsampling is not None:
            # downsample
            if z.ndim == 3:
                pool_fn = nn.functional.max_pool1d
            elif z.ndim == 4:
                pool_fn = nn.functional.max_pool2d
            elif z.ndim == 5:
                pool_fn = nn.functional.max_pool3d
            else:
                raise ValueError('Downsampling expects 1D, 2D or 3D input.')
            z = pool_fn(z, self.downsampling, ceil_mode=True)
        return z

    def forward(self, x1, x2):
        for blk in self.blocks:
            x1, x2 = blk(x1, x2)
        return x1, x2


class Concat_SS2Dv2(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 2
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear


        # in proj =======================================
        d_proj = self.d_inner
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.out_norm = LayerNorm(self.d_inner) 
       
        self.oact = False
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner*2, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = 2 * H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        x_fuse = ConcatScan_multimodal.apply(x, x2)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
      
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            x_fuse, dts, Bs, Cs = to_fp32(x_fuse, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, 2*H*W)
        
        y, y2 = ConcatMerge_multimodal.apply(ys)

        y = y.view(B, -1, H, W)
        y2 = y2.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
            y2 = y2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)
        y2 = out_norm(y2)

        return y.to(x.dtype), y2.to(x2.dtype)

    def forward(self, x: torch.Tensor,  x2: torch.Tensor):
        x = self.in_proj(x)
        x2 = self.in_proj(x2)
        
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
        x = self.act(x)
        x2 = self.act(x2)
        y1, y2 = self.forward_corev2(x, x2)
        y1 = self.out_act(y1)
        y2 = self.out_act(y2)
        
        b, d, h, w = x.shape
        x_squeeze = self.avg_pool(x).view(b, d)
        x2_squeeze = self.avg_pool(x2).view(b, d)
        x_exitation = self.fc1(x_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
        x2_exitation = self.fc1(x2_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() 
        y1 = y1 * x2_exitation
        y2 = y2 * x_exitation
        
        y = torch.concat([y1, y2], dim=-1)
        out = self.dropout(self.out_proj(y))
        return out


class ConcatFusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        **kwargs,
    ):
        super().__init__()

        self.norm = norm_layer(hidden_dim)
        self.concatSS2D = Concat_SS2Dv2(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dropout=attn_drop_rate, 
            **kwargs
        )
        
        self.drop_path = DropPath(drop_path)
        
    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x = self.drop_path(self.concatSS2D(x1_perm, x2_perm))
        
        x = x.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x_out = x1 + x2 + x
        return x_out


class ShallowFuse_SS2Dv2(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 2
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear


        # in proj =======================================
        d_proj = self.d_inner
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.out_norm = LayerNorm(self.d_inner) 
       
        self.oact = False
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = 2 * H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        x_fuse = ConcatScan_multimodal.apply(x, x2)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
      
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            x_fuse, dts, Bs, Cs = to_fp32(x_fuse, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, 2*H*W)
        
        y, y2 = ConcatMerge_multimodal.apply(ys)
        y = y.view(B, -1, H, W)
        y2 = y2.view(B, -1, H, W)
        
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
            y2 = y2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)
        y2 = out_norm(y2)

        return y.to(x.dtype), y2.to(x2.dtype)

    def forward(self, x: torch.Tensor,  x2: torch.Tensor):
        x = self.in_proj(x)
        x2 = self.in_proj(x2)
        
        if not self.channel_first:
            x_proj = x.permute(0, 3, 1, 2).contiguous()
            x2_proj = x2.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x_dconv = self.conv2d(x_proj) # (b, d, h, w)
            x2_dconv = self.conv2d(x2_proj) # (b, d, h, w)
        x_dconv = self.act(x_dconv)
        x2_dconv = self.act(x2_dconv)
       
        y1, y2 = self.forward_corev2(x_dconv, x2_dconv)
        
        y1 = self.out_act(y1)
        y2 = self.out_act(y2)
        
        b, d, h, w = x_proj.shape
        x_squeeze = self.avg_pool(x_proj).view(b, d)
        x2_squeeze = self.avg_pool(x2_proj).view(b, d)
        x_exitation = self.fc1(x_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
        x2_exitation = self.fc1(x2_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() 
        y1 = y1 * x2_exitation
        y2 = y2 * x_exitation
        
        out = self.dropout(self.out_proj(y1))
        out2 = self.dropout(self.out_proj(y2))
        
        return out, out2


class ShallowFuse_SS2Dv3(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 2
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear


        # in proj =======================================
        d_proj = self.d_inner
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.out_norm = LayerNorm(self.d_inner) 
       
        self.oact = False
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = 2 * H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        x_fuse = ConcatScan_multimodal.apply(x, x2)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
      
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            x_fuse, dts, Bs, Cs = to_fp32(x_fuse, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, 2*H*W)
        
        y, y2 = ConcatMerge_multimodal.apply(ys)

        y = y.view(B, -1, H, W)
        y2 = y2.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
            y2 = y2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)
        y2 = out_norm(y2)

        return y.to(x.dtype), y2.to(x2.dtype)

    def forward(self, x: torch.Tensor,  x2: torch.Tensor):
        x = self.in_proj(x)
        x2 = self.in_proj(x2)
        
        if not self.channel_first:
            x_proj = x.permute(0, 3, 1, 2).contiguous()
            x2_proj = x2.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x_dconv = self.conv2d(x_proj) # (b, d, h, w)
            x2_dconv = self.conv2d(x2_proj) # (b, d, h, w)
        x_dconv = self.act(x_dconv)
        x2_dconv = self.act(x2_dconv)
        y1, y2 = self.forward_corev2(x_dconv, x2_dconv)
        y1 = self.out_act(y1)
        y2 = self.out_act(y2)
        
        b, d, h, w = x_proj.shape
        x_avg_squeeze = self.avg_pool(x_proj).view(b, d)
        x_max_squeeze = self.max_pool(x_proj).view(b, d)
        x2_avg_squeeze = self.avg_pool(x2_proj).view(b, d)
        x2_max_squeeze = self.max_pool(x2_proj).view(b, d)
        x_avg_exitation = self.fc1(x_avg_squeeze)
        x_max_exitation = self.fc1(x_max_squeeze)
        x_channel_atten = self.sigmoid(x_avg_exitation + x_max_exitation).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
        x2_avg_exitation = self.fc1(x2_avg_squeeze)
        x2_max_exitation = self.fc1(x2_max_squeeze)
        x2_channel_atten = self.sigmoid(x2_avg_exitation + x2_max_exitation).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
        
        y1 = y1 * x2_channel_atten
        y2 = y2 * x_channel_atten
        
        out = self.dropout(self.out_proj(y1))
        out2 = self.dropout(self.out_proj(y2))
        return out, out2



class ShallowFuse_SS2Dv4(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 2
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear


        # in proj =======================================
        d_proj = self.d_inner
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.out_norm = LayerNorm(self.d_inner) 
       
        self.oact = False
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        x_swap = SwappingScan_multiview.apply(x, x2)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_swap, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
      
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        x_swap = x_swap.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            x_swap, dts, Bs, Cs = to_fp32(x_swap, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            x_swap, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H*W)
        
        y, y2 = SwappingMerge_multiview.apply(ys)
        y = y.view(B, -1, H, W)
        y2 = y2.view(B, -1, H, W)
        
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
            y2 = y2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)
        y2 = out_norm(y2)

        return y.to(x.dtype), y2.to(x2.dtype)

    def forward(self, x: torch.Tensor,  x2: torch.Tensor):
        x = self.in_proj(x)
        x2 = self.in_proj(x2)
        
        if not self.channel_first:
            x_proj = x.permute(0, 3, 1, 2).contiguous()
            x2_proj = x2.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x_dconv = self.conv2d(x_proj) # (b, d, h, w)
            x2_dconv = self.conv2d(x2_proj) # (b, d, h, w)
        x_dconv = self.act(x_dconv)
        x2_dconv = self.act(x2_dconv)
       
        y1, y2 = self.forward_corev2(x_dconv, x2_dconv)
        
        y1 = self.out_act(y1)
        y2 = self.out_act(y2)
        
        b, d, h, w = x_proj.shape
        x_squeeze = self.avg_pool(x_proj).view(b, d)
        x2_squeeze = self.avg_pool(x2_proj).view(b, d)
        x_exitation = self.fc1(x_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
        x2_exitation = self.fc1(x2_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() 
        y1 = y1 * x2_exitation
        y2 = y2 * x_exitation
        
        out = self.dropout(self.out_proj(y1))
        out2 = self.dropout(self.out_proj(y2))
        
        return out, out2


class ShallowFusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        **kwargs,
    ):
        super().__init__()

        self.norm = norm_layer(hidden_dim)
        self.shallowfuseSS2D = ShallowFuse_SS2Dv2(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dropout=attn_drop_rate, 
            **kwargs
        )
        
        self.drop_path = DropPath(drop_path)
        
    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x1_shallowfuse, x2_shallowfuse = self.drop_path(self.shallowfuseSS2D(x1_perm, x2_perm))
        
        x1_shallowfuse = x1_shallowfuse.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x2_shallowfuse = x2_shallowfuse.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        
        x1_out = x1 + x1_shallowfuse
        x2_out = x2 + x2_shallowfuse
        
        return x1_out, x2_out


class ShallowFusionBlock_v3(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        **kwargs,
    ):
        super().__init__()

        self.norm = norm_layer(hidden_dim)
        self.shallowfuseSS2D = ShallowFuse_SS2Dv3(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dropout=attn_drop_rate, 
            **kwargs
        )
        
        self.drop_path = DropPath(drop_path)
        
    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x1_shallowfuse, x2_shallowfuse = self.drop_path(self.shallowfuseSS2D(x1_perm, x2_perm))
        
        x1_shallowfuse = x1_shallowfuse.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x2_shallowfuse = x2_shallowfuse.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        
        x1_out = x1 + x1_shallowfuse
        x2_out = x2 + x2_shallowfuse
        
        return x1_out, x2_out


class ShallowFusionBlock_v4(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        **kwargs,
    ):
        super().__init__()

        self.norm = norm_layer(hidden_dim)
        self.shallowfuseSS2D = ShallowFuse_SS2Dv4(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dropout=attn_drop_rate, 
            **kwargs
        )
        
        self.drop_path = DropPath(drop_path)
        
    def forward(self, x1, x2):
        x1_norm = self.norm(x1)  # Shape: [B, C, H, W]
        x2_norm = self.norm(x2)  # Shape: [B, C, H, W]

        x1_perm = x1_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        x2_perm = x2_norm.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        x1_shallowfuse, x2_shallowfuse = self.drop_path(self.shallowfuseSS2D(x1_perm, x2_perm))
        
        x1_shallowfuse = x1_shallowfuse.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        x2_shallowfuse = x2_shallowfuse.permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
        
        x1_out = x1 + x1_shallowfuse
        x2_out = x2 + x2_shallowfuse
        
        return x1_out, x2_out


class SS2Dv2(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            def scan_rowcol(
                x: torch.Tensor, 
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # (2*c)
                _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width = True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight = self.x_proj_weight.view(4, -1, D)[:2].contiguous(), 
                proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3) # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight = self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
                proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
    

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v0",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2Dv2(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
        

class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=2, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v0",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.2, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v3", # "v1", "v2", "v3"
        patchembed_version: str = "v2", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        # =========================
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
# VSSM tiny: depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, SSM_RATIO: 1.0,  
# VSSM_small: depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, SSM_RATIO: 2.0
# VSSM_base: depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,  SSM_RATIO: 2.0
class Backbone_VSSM(VSSM):
    def __init__(self, depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, ssm_ratio=2.0,
                patch_size=4, in_chans=3, num_classes=1000, 
                ssm_d_state=1, ssm_dt_rank="auto", ssm_act_layer="silu",
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
                ssm_init="v0", forward_type="v05_noz", 
                mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                patch_norm=True, downsample_version="v3", patchembed_version="v2", 
                use_checkpoint=False, posembed=False, imgsize=224, 
                out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln2d", 
                **kwargs,
        ):
        super().__init__(depths=depths, dims=dims, drop_path_rate=drop_path_rate, 
                        patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, 
                        ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank, ssm_act_layer=ssm_act_layer,
                        ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias, ssm_drop_rate=ssm_drop_rate, 
                        ssm_init=ssm_init, forward_type=forward_type, 
                        mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer, mlp_drop_rate=mlp_drop_rate, gmlp=gmlp,
                        patch_norm=patch_norm, downsample_version=downsample_version, patchembed_version=patchembed_version, 
                        use_checkpoint=use_checkpoint, posembed=posembed, imgsize=imgsize, 
                        norm_layer=norm_layer,
                        **kwargs)      
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None) 
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        
        return outs


class Backbone_VSSM_DINO(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Store the expected output dimension for the DINO head
        self.embed_dim = backbone.dims[3]
        
    def forward(self, x):
        # Check and handle input shape
        if len(x.shape) == 3:
            # Already processed - unlikely but handling just in case
            B, N, C = x.shape
            return x
        
        # Get features from the backbone
        features = self.backbone(x)
        
        # Handle the output according to what backbone returns
        if isinstance(features, list):
            # It's returning a list of feature maps from different layers
            # Use the last one (highest level features)
            features = features[3]
        
        # Features may have shape [B, C, H, W] - apply global average pooling 
        # to get features of shape [B, C]
        if len(features.shape) == 4:
            features = features.mean(dim=[2, 3])
        
        return features


def single_channel_feature_extract_mamba(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=2, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


if __name__ ==  "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Backbone_VSSM(out_indices=(0, 1, 2, 3))
    model.to(device)
    
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  
    outputs = model.forward(input_tensor)
    
    for i, output in enumerate(outputs):
        print(f"Output from layer {i}: {output.shape}")