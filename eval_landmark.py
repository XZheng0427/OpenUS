"""Fetal-brain landmark training/eval on BrainBenchmark with OpenUS.
"""

import argparse
import contextlib
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import utils
from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2
from downstream_tasks.landmark.backbone_wrapper import VSSMFeatureExtractor
from downstream_tasks.landmark.head_landmark import build_head
from downstream_tasks.landmark.dataset_brainbench import (
    BrainBenchmarkLandmarkDataset, landmark_collate,
)
from downstream_tasks.landmark.transforms_landmark import (
    make_train_transform, make_val_transform,
)
from downstream_tasks.landmark.metrics_landmark import (
    soft_argmax_2d, rescale_to_original, mse_pixels, sdr_at,
)
from downstream_tasks.landmark.losses_landmark import make_loss, loss_expects_logits


# ---------- baselines ------------------------------------------------------

def _baselines_on_split(template_24x2: np.ndarray, recs, taus):
    """Compute train-mean and image-center baselines on one split's records.

    template_24x2: [24, 2] mean landmark template (in original-image pixel space).
    recs: list of manifest records, each with 'landmarks' [24, 2] and 'image_hw' [H, W].

    Returns: {"train_mean_template": {...}, "image_center": {...}}
        where each inner dict has 'mse' and 'sdr_<tau>' for each tau in taus.
        Returns {"train_mean_template": None, "image_center": None} when recs is empty
        (v7-A path: no val split).
    """
    if not recs:
        return {"train_mean_template": None, "image_center": None}
    gt = np.asarray([r["landmarks"] for r in recs], dtype=np.float64)        # [N, 24, 2]
    hw = np.asarray([r["image_hw"]  for r in recs], dtype=np.float64)        # [N, 2] = (H, W)

    def _metrics(pred):
        diff = pred - gt
        sq   = (diff ** 2).sum(axis=-1)
        err  = np.sqrt(sq)
        out  = {"mse": float(sq.mean())}
        for tau in taus:
            out[f"sdr_{tau}"] = float((err <= tau).mean())
        return out

    # train-mean template: same 24-coord vector for every image
    pred_template = np.broadcast_to(template_24x2[None, :, :], gt.shape).copy()

    # image-center: (W/2, H/2) per image, replicated across the 24 channels
    centers = np.stack([hw[:, 1] / 2.0, hw[:, 0] / 2.0], axis=-1)             # [N, 2]
    pred_center = np.broadcast_to(centers[:, None, :], gt.shape).copy()

    return {
        "train_mean_template": _metrics(pred_template),
        "image_center":        _metrics(pred_center),
    }


def compute_baselines(train_records, val_records, test_records, taus=(2.0, 4.0, 10.0)):
    """Compute trivial baselines across val and test splits.

    Returns:
        baselines: dict with shape
            {
              "train_mean_template": {"val": {...}, "test": {...}},
              "image_center":        {"val": {...}, "test": {...}},
              "template_24x2":       [[x1,y1], ...]   # the train-mean template, for record
            }
    """
    train_coords = np.asarray([r["landmarks"] for r in train_records], dtype=np.float64)
    template = train_coords.mean(axis=0)  # [24, 2]

    val_b  = _baselines_on_split(template, val_records,  taus)
    test_b = _baselines_on_split(template, test_records, taus)
    return {
        "train_mean_template": {"val": val_b["train_mean_template"],
                                "test": test_b["train_mean_template"]},
        "image_center":        {"val": val_b["image_center"],
                                "test": test_b["image_center"]},
        "template_24x2":       template.tolist(),
    }


def _print_baselines(baselines, args):
    print("========== BASELINES ==========")
    for bname in ("train_mean_template", "image_center"):
        print(f"  {bname}:")
        for split_name in ("val", "test"):
            m = baselines[bname][split_name]
            if m is None:
                # v7-A path: no val split.
                print(f"    {split_name:4s}  (no records)")
                continue
            sdr_str = "  ".join(f"sdr@{int(tau)}={m[f'sdr_{tau}']*100:5.2f}%"
                                for tau in args.sdr_taus)
            print(f"    {split_name:4s}  mse={m['mse']:11.2f}  {sdr_str}")


def _save_baselines_json(baselines, output_dir):
    path = os.path.join(output_dir, "baselines.json")
    # numpy lists are already JSON-safe (we converted template via .tolist())
    with open(path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"baselines saved to {path}")


# ---------- checkpoint loading ---------------------------------------------

def _load_openus(backbone: nn.Module, ckpt_path: str, key: str = "teacher"):
    """Load OpenUS checkpoint into backbone (strict=False with prefix-strip)."""
    print(f"loading OpenUS weights from {ckpt_path} (key={key!r})")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if key not in ckpt:
        raise SystemExit(f"checkpoint missing key {key!r}; have {list(ckpt)}")
    raw = ckpt[key]
    sd = {}
    for k, v in raw.items():
        if k.startswith("backbone."):
            sd[k[len("backbone."):]] = v
        elif k.startswith("module.backbone."):
            sd[k[len("module.backbone."):]] = v
    if not sd:
        raise SystemExit(
            f"no 'backbone.*' keys after prefix-strip in {ckpt_path}; "
            f"first 3 raw keys: {list(raw)[:3]}"
        )
    msg = backbone.load_state_dict(sd, strict=False)
    bad = [k for k in msg.missing_keys
           if k.startswith("patch_embed.") or k.startswith("layers.")]
    if bad:
        raise SystemExit(
            f"OpenUS load left encoder backbone keys missing — prefix-strip wrong. "
            f"first 5: {bad[:5]}"
        )
    print(f"  raw keys: {len(raw)}, after strip: {len(sd)}")
    print(f"  missing_keys ({len(msg.missing_keys)}): {msg.missing_keys[:5]}{'...' if len(msg.missing_keys) > 5 else ''}")
    print(f"  unexpected_keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:5]}{'...' if len(msg.unexpected_keys) > 5 else ''}")


# ---------- distributed metric reduction -----------------------------------

def _reduce_scalars(values: dict) -> dict:
    """All-reduce a dict of {name: float} across ranks; returns mean."""
    if not (dist.is_available() and dist.is_initialized()):
        return values
    world = dist.get_world_size()
    if world <= 1:
        return values
    tensor = torch.tensor(list(values.values()), dtype=torch.float64,
                          device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world
    return {k: float(tensor[i].item()) for i, k in enumerate(values)}


# ---------- one-epoch train/eval -------------------------------------------

def train_one_epoch(extractor, head, loader, optimizer, criterion, epoch, args):
    extractor.eval()
    head.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Train [{epoch}]"
    use_coord_loss = args.coord_loss_weight > 0.0
    for imgs, heatmaps, coords_in, _meta in metric_logger.log_every(loader, args.log_freq, header):
        imgs      = imgs.cuda(non_blocking=True)
        heatmaps  = heatmaps.cuda(non_blocking=True)
        coords_in = coords_in.cuda(non_blocking=True)   # [B, K, 2] in input-resolution space

        ctx = torch.no_grad() if args.freeze_backbone_fully else contextlib.nullcontext()
        with ctx:
            feats = extractor(imgs)

        # lq=imgs is consumed by LandmarkUNetHead's stem; ignored by other heads.
        pred = head(feats, orig_hw=imgs.shape[2:], lq=imgs)
        hm_loss = criterion(pred, heatmaps)

        # Soft-argmax interprets its input as a probability-like score map;
        # for BCE/focal we sigmoid first so the beta-temperature is comparable
        # across loss families. MSE keeps the v4 raw-logits path.
        pred_for_decode = torch.sigmoid(pred) if loss_expects_logits(args.loss_type) else pred

        if use_coord_loss:
            pred_coords = soft_argmax_2d(pred_for_decode, beta=args.soft_argmax_beta)
            # Normalize coord error by img_size so the L1 magnitude is O(0.01–0.1)
            # regardless of resolution; tune via --coord_loss_weight.
            coord_loss = (pred_coords - coords_in).abs().mean() / args.img_size
            loss = hm_loss + args.coord_loss_weight * coord_loss
        else:
            coord_loss = torch.zeros((), device=hm_loss.device)
            loss = hm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item(),
                             hm_loss=hm_loss.item(),
                             coord_loss=coord_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: float(m.global_avg) for k, m in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(extractor, head, loader, criterion, args, header="Eval"):
    extractor.eval()
    head.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    total_sq_err = 0.0   # sum over (image, landmark) pairs
    total_correct = {tau: 0.0 for tau in args.sdr_taus}
    n_pairs = 0

    for imgs, heatmaps, _coords_in, meta in metric_logger.log_every(loader, args.log_freq, header):
        imgs     = imgs.cuda(non_blocking=True)
        heatmaps = heatmaps.cuda(non_blocking=True)
        gt_orig  = meta["orig_coords"].cuda(non_blocking=True)   # [B, K, 2]
        scale    = meta["scale"].cuda(non_blocking=True)         # [B, 2]

        # lq=imgs is consumed by LandmarkUNetHead's stem; ignored by other heads.
        pred = head(extractor(imgs), orig_hw=imgs.shape[2:], lq=imgs)
        loss = criterion(pred, heatmaps)

        pred_for_decode = torch.sigmoid(pred) if loss_expects_logits(args.loss_type) else pred
        pred_coords_in = soft_argmax_2d(pred_for_decode, beta=args.soft_argmax_beta)
        pred_orig = rescale_to_original(pred_coords_in, scale)

        diff = pred_orig - gt_orig                          # [B, K, 2]
        sq   = (diff ** 2).sum(dim=-1)                      # [B, K]
        err  = torch.sqrt(sq)
        B, K = sq.shape
        n_pairs += B * K
        total_sq_err += sq.sum().item()
        for tau in args.sdr_taus:
            total_correct[tau] += (err <= tau).float().sum().item()

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    if n_pairs == 0:
        return {"loss": 0.0, "mse": float("inf"),
                **{f"sdr_{tau}": 0.0 for tau in args.sdr_taus}}

    # accumulate across ranks (sum of squared errors + sum of correct)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.tensor(
            [total_sq_err, n_pairs] + [total_correct[tau] for tau in args.sdr_taus],
            dtype=torch.float64, device=torch.cuda.current_device(),
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_sq_err = t[0].item()
        n_pairs      = int(t[1].item())
        for i, tau in enumerate(args.sdr_taus):
            total_correct[tau] = t[2 + i].item()

    return {
        "loss": metric_logger.meters["loss"].global_avg,
        "mse":  total_sq_err / max(n_pairs, 1),
        **{f"sdr_{tau}": total_correct[tau] / max(n_pairs, 1) for tau in args.sdr_taus},
    }


# ---------- main -----------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)

    # ---------- data ----------
    train_tf = make_train_transform(
        img_size=args.img_size,
        sigma=args.sigma,
        enable_flips=args.enable_flips,
        enable_jitter=args.enable_jitter,
    )
    val_tf = make_val_transform(img_size=args.img_size, sigma=args.sigma)

    train_ds = BrainBenchmarkLandmarkDataset(
        manifest_path=args.landmark_manifest,
        images_root=args.images_root,
        split="train",
        transform=train_tf,
    )
    # v7-A path: --skip_val means the manifest has no val records (and
    # BrainBenchmarkLandmarkDataset raises on empty splits). Skip the
    # constructor entirely and treat val_ds as None / [].
    if args.skip_val:
        val_ds = None
        val_records = []
    else:
        val_ds = BrainBenchmarkLandmarkDataset(
            manifest_path=args.landmark_manifest,
            images_root=args.images_root,
            split="val",
            transform=val_tf,
        )
        val_records = val_ds.records
    test_ds = BrainBenchmarkLandmarkDataset(
        manifest_path=args.landmark_manifest,
        images_root=args.images_root,
        split="test",
        transform=val_tf,
    )
    val_size_str = f"{len(val_ds)}" if val_ds is not None else "0 (skip_val)"
    print(f"data loaded: train={len(train_ds)}  val={val_size_str}  test={len(test_ds)}")

    # ---------- baselines (computed once, before training) ----------
    baselines = compute_baselines(
        train_ds.records, val_records, test_ds.records,
        taus=tuple(args.sdr_taus),
    )
    if utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        _print_baselines(baselines, args)
        _save_baselines_json(baselines, args.output_dir)

    if utils.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size_per_gpu, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, collate_fn=landmark_collate, drop_last=False,
    )
    if args.skip_val:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size_per_gpu, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=landmark_collate, drop_last=False,
        )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size_per_gpu, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=landmark_collate, drop_last=False,
    )

    # ---------- backbone (frozen by default; selective per-stage unfreeze) ----------
    backbone = Backbone_DINOv2_VSSM_2(pretrained=args.pretrained_vmamba_init)
    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        _load_openus(backbone, args.pretrained_weights, key=args.checkpoint_key)
    else:
        print(f"WARNING: no OpenUS weights loaded ({args.pretrained_weights!r}); "
              f"backbone uses ImageNet init only")

    # Default: freeze everything (v4 path).
    for p in backbone.parameters():
        p.requires_grad = False

    # Selectively unfreeze whole stages and zero their stochastic-depth so
    # fine-tuning on a 55-image set isn't drowned in DropPath noise. v4 default
    # (--unfreeze_stages []) leaves everything frozen and reproduces v4 exactly.
    unfreeze_stages = sorted(set(args.unfreeze_stages or []))
    for i in unfreeze_stages:
        if not (0 <= i < len(backbone.layers)):
            raise ValueError(
                f"--unfreeze_stages index {i} out of range; backbone has "
                f"{len(backbone.layers)} stages (0..{len(backbone.layers) - 1})"
            )
        for p in backbone.layers[i].parameters():
            p.requires_grad = True
        for block in backbone.layers[i].blocks:
            dp = getattr(block, "drop_path", None)
            if dp is not None and hasattr(dp, "drop_prob"):
                dp.drop_prob = 0.0

    args.freeze_backbone_fully = len(unfreeze_stages) == 0
    extractor = VSSMFeatureExtractor(backbone).cuda().eval()
    if not args.freeze_backbone_fully:
        n_backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"unfrozen backbone stages: {unfreeze_stages}  "
              f"trainable backbone params: {n_backbone_trainable:,}")

    # ---------- head ----------
    head = build_head(
        head_type=args.head_type,
        num_landmarks=args.num_landmarks,
        backbone_dims=tuple(backbone.dims),
        img_size=args.img_size,
        patch_size=args.patch_size,
        linear_head_stage=args.linear_head_stage,
    ).cuda()
    n_head_params = sum(p.numel() for p in head.parameters())
    print(f"head trainable params: {n_head_params:,}  (head_type={args.head_type}"
          + (f", linear_head_stage={args.linear_head_stage}" if args.head_type == "linear" else "")
          + ")")

    if utils.get_world_size() > 1:
        head = DDP(head, device_ids=[args.gpu])
    head_module = head.module if isinstance(head, DDP) else head

    # ---------- optim ----------
    param_groups = [{
        "params": list(head.parameters()),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "name": "head",
    }]
    backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
    if backbone_trainable:
        param_groups.append({
            "params": backbone_trainable,
            "lr": args.lr * args.backbone_lr_scale,
            "weight_decay": args.weight_decay,
            "name": "backbone",
        })
        print(f"optimizer param groups: head={sum(p.numel() for p in head.parameters()):,} @ lr={args.lr}; "
              f"backbone={sum(p.numel() for p in backbone_trainable):,} @ lr={args.lr * args.backbone_lr_scale}")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )
    criterion = make_loss(
        args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_beta=args.focal_beta,
        focal_gamma=args.focal_gamma,
    )
    print(f"loss: {args.loss_type}"
          + (f" (alpha={args.focal_alpha} beta={args.focal_beta} gamma={args.focal_gamma})"
             if args.loss_type == "focal" else ""))

    # ---------- output dir / log file ----------
    if utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.output_dir, "log.txt")
    best_ckpt_path = os.path.join(args.output_dir, "head_best_val.pth")
    best_sdr_ckpt_path = os.path.join(args.output_dir, "head_best_val_sdr.pth")
    # v7-A path: no val carve-out → save only at the last epoch.
    final_ckpt_path = os.path.join(args.output_dir, "head_final.pth")
    # SDR is tracked at the largest tau in args.sdr_taus (10.0 in the default
    # protocol — the headline landmark metric and the most permissive, so
    # the criterion is non-zero earliest in training).
    sdr_tau_key = f"sdr_{max(args.sdr_taus)}"

    # ---------- resume ----------
    to_restore = {"epoch": 0, "best_val_mse": float("inf"), "best_val_sdr": 0.0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=head_module,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch  = to_restore["epoch"]
    best_val_mse = to_restore["best_val_mse"]
    best_val_sdr = to_restore["best_val_sdr"]

    def _trainable_backbone_state():
        """State-dict slice for the stages with requires_grad=True. Empty
        dict when the backbone is fully frozen (v4 path) so the checkpoint
        is byte-equivalent to v4."""
        if args.freeze_backbone_fully:
            return {}
        prefixes = tuple(f"layers.{i}." for i in unfreeze_stages)
        return {k: v for k, v in backbone.state_dict().items() if k.startswith(prefixes)}

    def _load_trainable_backbone_state(ck: dict):
        sd = ck.get("backbone_state_dict") or {}
        if not sd:
            return
        # Per-stage load. We deliberately avoid backbone.load_state_dict(strict=False)
        # because the VSSM blocks' custom _load_from_state_dict
        # (vmamba_models/dino_vmamba.py) unconditionally accesses
        # state_dict[prefix + "weight"] for a reshape; with a partial
        # state dict the frozen-stage keys aren't present and it raises
        # KeyError before strict=False has a chance to be permissive.
        # Loading each unfrozen stage's sub-module directly sidesteps the
        # frozen modules entirely.
        for i in unfreeze_stages:
            prefix = f"layers.{i}."
            stage_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not stage_sd:
                continue
            backbone.layers[i].load_state_dict(stage_sd, strict=True)

    # ---------- epoch loop ----------
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            extractor, head, train_loader, optimizer, criterion, epoch, args,
        )
        scheduler.step()

        if args.skip_val:
            # v7-A path: no val evaluation, no per-epoch checkpoint saves.
            if utils.is_main_process():
                log = {"epoch": epoch,
                       **{f"train_{k}": v for k, v in train_stats.items()}}
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
                print(f"epoch {epoch}  train_loss={train_stats.get('loss'):.5f}  "
                      f"(skip_val; final ckpt at epoch {args.epochs - 1})")
        elif epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_stats = evaluate(extractor, head, val_loader, criterion, args,
                                 header=f"Val [{epoch}]")
            improved = val_stats["mse"] < best_val_mse
            sdr_improved = val_stats.get(sdr_tau_key, 0.0) > best_val_sdr
            if improved:
                best_val_mse = val_stats["mse"]
                if utils.is_main_process():
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict": head_module.state_dict(),
                        "backbone_state_dict": _trainable_backbone_state(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_mse": best_val_mse,
                        "args": vars(args),
                    }, best_ckpt_path)
            if sdr_improved:
                best_val_sdr = val_stats[sdr_tau_key]
                if utils.is_main_process():
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict": head_module.state_dict(),
                        "backbone_state_dict": _trainable_backbone_state(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_sdr": best_val_sdr,
                        "best_val_sdr_tau_key": sdr_tau_key,
                        "args": vars(args),
                    }, best_sdr_ckpt_path)

            if utils.is_main_process():
                log = {"epoch": epoch,
                       **{f"train_{k}": v for k, v in train_stats.items()},
                       **{f"val_{k}":   v for k, v in val_stats.items()},
                       "best_val_mse": best_val_mse,
                       "best_val_sdr": best_val_sdr,
                       "improved": improved,
                       "sdr_improved": sdr_improved}
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
                print(f"epoch {epoch}  train_loss={train_stats.get('loss'):.5f}  "
                      f"val_mse={val_stats['mse']:.3f}  "
                      f"val_sdr_4={val_stats.get('sdr_4.0', 0.0):.3f}  "
                      f"best_val_mse={best_val_mse:.3f}{'  *' if improved else ''}  "
                      f"best_val_sdr={best_val_sdr:.4f}{'  +' if sdr_improved else ''}")

    # v7-A: save the final-epoch checkpoint after the loop completes.
    if args.skip_val and utils.is_main_process():
        torch.save({
            "epoch": args.epochs,
            "state_dict": head_module.state_dict(),
            "backbone_state_dict": _trainable_backbone_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }, final_ckpt_path)
        print(f"saved final-epoch checkpoint to {final_ckpt_path}")

    # ---------- final test eval with both best-val checkpoints ----------
    def _final_test_with(ckpt_path, label, log_key):
        """Load ckpt, run test eval, print + return the metrics block."""
        if utils.is_main_process():
            print(f"\nfinal test eval with {label} checkpoint: {ckpt_path}")
        if os.path.isfile(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            msg = head_module.load_state_dict(ck["state_dict"], strict=True)
            _load_trainable_backbone_state(ck)
            if utils.is_main_process():
                if "best_val_mse" in ck:
                    print(f"  loaded epoch={ck['epoch']}  best_val_mse={ck['best_val_mse']:.3f}  "
                          f"msg={msg}")
                else:
                    print(f"  loaded epoch={ck['epoch']}  best_val_sdr={ck.get('best_val_sdr', 0):.4f}  "
                          f"msg={msg}")
        else:
            if utils.is_main_process():
                print(f"  WARNING: no checkpoint at {ckpt_path}; using current head weights")

        ts = evaluate(extractor, head, test_loader, criterion, args,
                      header=f"Test ({label})")
        if utils.is_main_process():
            tmpl_test = baselines["train_mean_template"]["test"]
            cntr_test = baselines["image_center"]["test"]
            gap_mse   = ts["mse"] - tmpl_test["mse"]
            print(f"\n========== FINAL TEST METRICS ({label}) ==========")
            for k, v in ts.items():
                print(f"  {k:12s} = {v:.5f}")
            print()
            print(f"========== MODEL vs BASELINES (test, {label}) ==========")
            print(f"  {'metric':<20s} {'model':>12s} {'train_mean':>12s} {'image_center':>14s} {'gap(M-T)':>12s}")
            print(f"  {'mse':<20s} {ts['mse']:>12.2f} {tmpl_test['mse']:>12.2f} {cntr_test['mse']:>14.2f} {gap_mse:>+12.2f}")
            for tau in args.sdr_taus:
                key = f"sdr_{tau}"
                print(f"  {key:<20s} "
                      f"{ts.get(key, 0.0)*100:>11.2f}% "
                      f"{tmpl_test[key]*100:>11.2f}% "
                      f"{cntr_test[key]*100:>13.2f}% "
                      f"{(ts.get(key, 0.0) - tmpl_test[key])*100:>+11.2f}%")
            verdict = "BEATS template" if gap_mse < 0 else "TIES OR LOSES to template"
            print(f"  → {verdict} (negative gap = model better)")

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    log_key: ts,
                    "best_val_mse": best_val_mse,
                    "best_val_sdr": best_val_sdr,
                    "baseline_test_train_mean":   tmpl_test,
                    "baseline_test_image_center": cntr_test,
                    f"{log_key}_model_minus_template_mse": gap_mse,
                }) + "\n")
        return ts

    if args.skip_val:
        # v7-A path: single final-epoch checkpoint, single test eval.
        # Log key "final_test" kept for `_aggregate_multiseed.py` compatibility.
        _final_test_with(final_ckpt_path, "final-epoch", "final_test")
    else:
        # Original best-MSE checkpoint (v4 protocol). Log key kept as "final_test"
        # for backward compatibility with `_aggregate_multiseed.py`.
        _final_test_with(best_ckpt_path,     "best-val-MSE", "final_test")
        # New best-SDR checkpoint (v6-A). Separate log record so the aggregator
        # can be extended to read it, while the original "final_test" key stays
        # untouched.
        _final_test_with(best_sdr_ckpt_path, "best-val-SDR", "final_test_best_sdr")


# ---------- argparse -------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser("OpenUS landmark eval on BrainBenchmark")

    # data
    p.add_argument("--landmark_manifest", required=True, type=str,
                   help="path to landmark_manifest.json (NOT manifest.json)")
    p.add_argument("--images_root", required=True, type=str,
                   help="dir to prepend to record['image'] paths")
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--sigma", default=8.0, type=float,
                   help="Gaussian heatmap sigma in pixels. Default 8 (working "
                        "range for 224-input). Values around 2 collapse to "
                        "the all-zeros local minimum with vanilla MSE.")
    p.add_argument("--coord_loss_weight", default=0.1, type=float,
                   help="Weight of the soft-argmax coord-domain L1 auxiliary loss. "
                        "0 disables it; values around 0.1 typically rescue the "
                        "heatmap-MSE collapse failure mode.")
    p.add_argument("--enable_flips", default=False, type=utils.bool_flag,
                   help="enable h/v flip augmentations (requires correct "
                        "HFLIP_PERM/VFLIP_PERM in landmark_schema.py)")
    p.add_argument("--enable_jitter", default=True, type=utils.bool_flag)
    p.add_argument("--num_landmarks", default=24, type=int)

    # arch / weights
    p.add_argument("--arch", default="vmamba_small", type=str,
                   choices=["vmamba_small"])
    p.add_argument("--pretrained_vmamba", default=True, type=utils.bool_flag,
                   help="kept for symmetry with eval_segmentation.py; effectively always True")
    p.add_argument("--pretrained_vmamba_init", required=True, type=str,
                   help="ImageNet VMamba checkpoint (e.g. vssm_small_0229_ckpt_epoch_222.pth)")
    p.add_argument("--pretrained_weights", default="", type=str,
                   help="OpenUS checkpoint (loaded on top of vmamba init)")
    p.add_argument("--checkpoint_key", default="teacher", type=str,
                   choices=["teacher", "student"])
    p.add_argument("--patch_size", default=4, type=int)
    p.add_argument("--head_type", default="mamba_decoder", type=str,
                   choices=["mamba_decoder", "linear", "unet"],
                   help="head type")
    p.add_argument("--linear_head_stage", default=0, type=int, choices=[0, 1, 2, 3],
                   help="When --head_type=linear, which VSSM feature map to "
                        "project from. 0 = highest resolution (56x56 for 224 input), "
                        "3 = lowest resolution / deepest semantic (7x7).")

    # when no new flags are passed.
    p.add_argument("--loss_type", default="mse", type=str,
                   choices=["mse", "bce", "focal"],
                   help="Heatmap loss family.")
    p.add_argument("--focal_alpha", default=2.0, type=float,
                   help="focal positives exponent on (1 - p).")
    p.add_argument("--focal_beta", default=4.0, type=float,
                   help="focal negatives weight exponent on (1 - target).")
    p.add_argument("--focal_gamma", default=2.0, type=float,
                   help="focal negatives modulator exponent on p.")

    p.add_argument("--unfreeze_stages", default=[], nargs="*", type=int,
                   help="VSSM stage indices to unfreeze (subset of {0,1,2,3}). "
                        "Empty (default) = full freeze.")
    p.add_argument("--backbone_lr_scale", default=0.1, type=float,
                   help="Multiplier applied to --lr for the unfrozen backbone "
                        "param group. Standard recipe: lr/10. Ignored when "
                        "--unfreeze_stages is empty.")

    p.add_argument("--skip_val", default=False, type=utils.bool_flag,
                   help="train without a val split (no early-stopping).")

    # optim
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--batch_size_per_gpu", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--lr_min", default=1e-6, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--val_freq", default=1, type=int)
    p.add_argument("--log_freq", default=20, type=int)

    # metrics
    p.add_argument("--sdr_taus", nargs="+", type=float, default=[2.0, 4.0, 10.0])
    p.add_argument("--soft_argmax_beta", default=100.0, type=float)

    # bookkeeping
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--load_from", default=None, type=str,
                   help="checkpoint to resume training from (relative to output_dir)")
    p.add_argument("--dist_url", default="env://", type=str)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
