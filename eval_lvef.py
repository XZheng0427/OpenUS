"""CAMUS LVEF regression training/eval with the OpenUS encoder.

"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import utils
from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2

from downstream_tasks.lvef.dataset_camus import (
    CAMUSLVEFDataset, lvef_collate, FRAME_KEYS,
)
from downstream_tasks.lvef.transforms_lvef import (
    make_train_transform, make_val_transform,
)
from downstream_tasks.lvef.metrics_lvef import RegressionAccumulator
from downstream_tasks.lvef.backbone_wrapper import VSSMFeatureExtractor
from downstream_tasks.lvef.head_lvef import build_head

# Reuse the OpenUS-DINO checkpoint loader from the landmark pipeline so the
# prefix-strip and warnings are bit-identical.
from eval_landmark import _load_openus


# ---------- helpers --------------------------------------------------------

def _make_criterion(loss_type: str) -> nn.Module:
    if loss_type == "l1":
        return nn.L1Loss(reduction="mean")
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction="mean", beta=1.0)
    if loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    raise ValueError(f"unknown loss_type {loss_type!r}")


def _encode_4view(extractor: nn.Module, frames: torch.Tensor):
    """Encode [B, V, 3, H, W] in a single forward; return list of per-view feature lists."""
    B, V, C, H, W = frames.shape
    flat = frames.reshape(B * V, C, H, W)
    feats_flat = extractor(flat)
    per_view = []
    for v in range(V):
        idx = torch.arange(v, B * V, V, device=feats_flat[0].device)
        view_feats = [f.index_select(0, idx) for f in feats_flat]
        per_view.append(view_feats)
    return per_view


# ---------- one-epoch train / eval ----------------------------------------

def train_one_epoch(extractor, head, loader, optimizer, criterion, epoch, args):
    extractor.eval()
    head.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Train [{epoch}]"
    for frames, efs, _meta in metric_logger.log_every(loader, args.log_freq, header):
        frames = frames.cuda(non_blocking=True)
        efs    = efs.cuda(non_blocking=True)

        with torch.no_grad():
            feats_per_view = _encode_4view(extractor, frames)

        pred = head(feats_per_view).squeeze(-1)
        loss = criterion(pred, efs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item(),
                             mae=(pred.detach() - efs).abs().mean().item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: float(m.global_avg) for k, m in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(extractor, head, loader, criterion, args, header="Eval"):
    extractor.eval()
    head.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    acc = RegressionAccumulator(keep_per_patient=True)

    for frames, efs, meta in metric_logger.log_every(loader, args.log_freq, header):
        frames = frames.cuda(non_blocking=True)
        efs    = efs.cuda(non_blocking=True)

        feats_per_view = _encode_4view(extractor, frames)
        pred = head(feats_per_view).squeeze(-1)
        loss = criterion(pred, efs)

        acc.update(pred, efs, patients=meta["patient"])
        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    acc.all_reduce()
    metrics = acc.metrics()
    return {
        "loss": metric_logger.meters["loss"].global_avg,
        **metrics,
    }, acc


def _baselines(train_records, split_records):
    if not split_records:
        return None
    train_efs = [r["ef"] for r in train_records]
    mu = sum(train_efs) / len(train_efs)
    med = sorted(train_efs)[len(train_efs) // 2]
    gt = [r["ef"] for r in split_records]
    def _mae(pred):
        return sum(abs(p - g) for p, g in zip(pred, gt)) / len(gt)
    return {
        "train_mean": {"pred": mu,  "mae": _mae([mu]  * len(gt))},
        "train_median": {"pred": med, "mae": _mae([med] * len(gt))},
        "constant_50": {"pred": 50, "mae": _mae([50.0] * len(gt))},
    }


def _print_baselines(baselines):
    print("========== BASELINES ==========")
    for split, b in baselines.items():
        if b is None:
            print(f"  {split:4s}  (no records)")
            continue
        print(f"  {split}:")
        for name, m in b.items():
            print(f"    {name:<12s} pred={m['pred']:>5.2f}  MAE={m['mae']:>5.3f}")


def _write_per_patient_csv(path: str, acc: RegressionAccumulator) -> None:
    if acc.per_patient_patient is None:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "gt_ef", "pred_ef", "abs_err"])
        for p, g, pr, e in zip(acc.per_patient_patient,
                               acc.per_patient_gt,
                               acc.per_patient_pred,
                               acc.per_patient_abs):
            w.writerow([p, f"{g:.4f}", f"{pr:.4f}", f"{e:.4f}"])


# ---------- main -----------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)

    train_tf = make_train_transform(
        img_size=args.img_size,
        enable_vflip=args.enable_vflip,
        enable_jitter=args.enable_jitter,
    )
    val_tf = make_val_transform(img_size=args.img_size)

    train_ds = CAMUSLVEFDataset(args.camus_manifest, args.images_root, "train", train_tf)
    val_ds   = (None if args.skip_val else
                CAMUSLVEFDataset(args.camus_manifest, args.images_root, "val", val_tf))
    test_ds  = CAMUSLVEFDataset(args.camus_manifest, args.images_root, "test", val_tf)
    print(f"data loaded: train={len(train_ds)}  "
          f"val={'0 (skip_val)' if val_ds is None else len(val_ds)}  "
          f"test={len(test_ds)}")

    if utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        baselines = {
            "val":  _baselines(train_ds.records, [] if val_ds is None else val_ds.records),
            "test": _baselines(train_ds.records, test_ds.records),
        }
        _print_baselines(baselines)
        with open(os.path.join(args.output_dir, "baselines.json"), "w") as f:
            json.dump(baselines, f, indent=2)

    if utils.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size_per_gpu, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, collate_fn=lvef_collate, drop_last=False,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=args.batch_size_per_gpu, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lvef_collate, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size_per_gpu, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lvef_collate, drop_last=False,
    )

    # ---------- backbone (frozen) ----------
    backbone = Backbone_DINOv2_VSSM_2(pretrained=args.pretrained_vmamba_init)
    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        _load_openus(backbone, args.pretrained_weights, key=args.checkpoint_key)
    else:
        print(f"WARNING: no OpenUS weights loaded ({args.pretrained_weights!r}); "
              "backbone uses ImageNet init only")

    for p in backbone.parameters():
        p.requires_grad = False
    extractor = VSSMFeatureExtractor(backbone).cuda().eval()

    deepest_dim = int(backbone.dims[-1])
    head = build_head(
        backbone=args.arch,
        n_views=len(FRAME_KEYS),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        in_dim_per_view=deepest_dim,
    ).cuda()
    n_head_params = sum(p.numel() for p in head.parameters())
    print(f"head trainable params: {n_head_params:,}  "
          f"(in_dim_per_view={deepest_dim}, hidden={args.hidden_dim})")

    if utils.get_world_size() > 1:
        head = DDP(head, device_ids=[args.gpu])
    head_module = head.module if isinstance(head, DDP) else head

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )
    criterion = _make_criterion(args.loss)
    print(f"loss: {args.loss}")

    log_path = os.path.join(args.output_dir, "log.txt")
    best_ckpt_path = os.path.join(args.output_dir, "head_best_val.pth")
    final_ckpt_path = os.path.join(args.output_dir, "head_final.pth")

    to_restore = {"epoch": 0, "best_val_mae": float("inf")}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=head_module,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch  = to_restore["epoch"]
    best_val_mae = to_restore["best_val_mae"]

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            extractor, head, train_loader, optimizer, criterion, epoch, args,
        )
        scheduler.step()

        if args.skip_val:
            if utils.is_main_process():
                log = {"epoch": epoch,
                       **{f"train_{k}": v for k, v in train_stats.items()}}
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
                print(f"epoch {epoch}  train_loss={train_stats.get('loss'):.5f}  "
                      f"train_mae={train_stats.get('mae', 0.0):.3f}  "
                      f"(skip_val; final ckpt at epoch {args.epochs - 1})")
        elif epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_stats, _ = evaluate(extractor, head, val_loader, criterion, args,
                                    header=f"Val [{epoch}]")
            improved = val_stats["mae"] < best_val_mae
            if improved:
                best_val_mae = val_stats["mae"]
                if utils.is_main_process():
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict": head_module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_mae": best_val_mae,
                        "args": vars(args),
                    }, best_ckpt_path)
            if utils.is_main_process():
                log = {"epoch": epoch,
                       **{f"train_{k}": v for k, v in train_stats.items()},
                       **{f"val_{k}":   v for k, v in val_stats.items()},
                       "best_val_mae": best_val_mae,
                       "improved": improved}
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
                print(f"epoch {epoch}  train_loss={train_stats.get('loss'):.5f}  "
                      f"val_mae={val_stats['mae']:.3f}  "
                      f"val_rmse={val_stats.get('rmse', 0.0):.3f}  "
                      f"best_val_mae={best_val_mae:.3f}{'  *' if improved else ''}")

    if args.skip_val and utils.is_main_process():
        torch.save({
            "epoch": args.epochs,
            "state_dict": head_module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }, final_ckpt_path)
        print(f"saved final-epoch checkpoint to {final_ckpt_path}")

    test_ckpt_path = final_ckpt_path if args.skip_val else best_ckpt_path
    test_ckpt_label = "final-epoch" if args.skip_val else "best-val"
    if utils.is_main_process():
        print(f"\nfinal test eval with {test_ckpt_label} checkpoint: {test_ckpt_path}")
    if os.path.isfile(test_ckpt_path):
        ck = torch.load(test_ckpt_path, map_location="cpu", weights_only=False)
        msg = head_module.load_state_dict(ck["state_dict"], strict=True)
        if utils.is_main_process():
            tag = (f"best_val_mae={ck['best_val_mae']:.3f}"
                   if "best_val_mae" in ck else f"epoch={ck['epoch']}")
            print(f"  loaded {tag}  msg={msg}")
    else:
        if utils.is_main_process():
            print(f"  WARNING: no checkpoint at {test_ckpt_path}; using current head")

    test_stats, test_acc = evaluate(extractor, head, test_loader, criterion, args,
                                    header="Test (final)")

    if utils.is_main_process():
        print(f"\n========== FINAL TEST METRICS ==========")
        for k, v in test_stats.items():
            if isinstance(v, (int, float)):
                print(f"  {k:14s} = {v:.5f}")

        csv_path = os.path.join(args.output_dir, "test_per_patient.csv")
        _write_per_patient_csv(csv_path, test_acc)
        print(f"  per-patient predictions -> {csv_path}")

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "final_test": {k: v for k, v in test_stats.items()
                               if isinstance(v, (int, float))},
                "best_val_mae": None if args.skip_val else best_val_mae,
            }) + "\n")


# ---------- argparse -------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser("OpenUS LVEF regression on CAMUS")

    # data
    p.add_argument("--camus_manifest", required=True, type=str)
    p.add_argument("--images_root",    required=True, type=str)
    p.add_argument("--img_size",     default=224, type=int)
    p.add_argument("--enable_vflip", default=False, type=utils.bool_flag)
    p.add_argument("--enable_jitter", default=True, type=utils.bool_flag)

    # arch / weights
    p.add_argument("--arch", default="vmamba_small", choices=["vmamba_small"], type=str)
    p.add_argument("--pretrained_vmamba_init", required=True, type=str,
                   help="ImageNet VMamba checkpoint (e.g. vssm_small_0229_ckpt_epoch_222.pth)")
    p.add_argument("--pretrained_weights", default="", type=str,
                   help="OpenUS checkpoint (loaded on top of vmamba init)")
    p.add_argument("--checkpoint_key", default="teacher", choices=["teacher", "student"], type=str)
    p.add_argument("--hidden_dim", default=512, type=int)
    p.add_argument("--dropout",    default=0.1, type=float)

    # optim
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch_size_per_gpu", default=16, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--lr_min", default=1e-6, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--val_freq", default=1, type=int)
    p.add_argument("--log_freq", default=20, type=int)
    p.add_argument("--loss", default="smooth_l1", choices=["l1", "smooth_l1", "mse"], type=str)

    # protocol
    p.add_argument("--skip_val", default=False, type=utils.bool_flag)

    # bookkeeping
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--load_from", default=None, type=str)
    p.add_argument("--dist_url", default="env://", type=str)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
