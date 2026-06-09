"""OpenUS downstream task 4: train / eval Rotated Faster R-CNN with the
OpenUS backbone on FOCUS (fetal cardiac four-chamber view).

"""

import argparse
import json
import os
import sys
from pathlib import Path


# ---------- OpenUS repo path -----------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def _bool(s):
    s = s.lower()
    if s in ("true", "1", "yes", "y"): return True
    if s in ("false", "0", "no", "n"): return False
    raise argparse.ArgumentTypeError(f"bool flag expected, got {s!r}")


# ---------- CLI -------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        "OpenUS cardiac-detection eval on FOCUS (Rotated Faster R-CNN)"
    )

    # data
    p.add_argument(
        "--data_root", required=True, type=str,
        help="FOCUS-dataset directory containing training/ validation/ testing/",
    )
    p.add_argument(
        "--prepared_dir", required=True, type=str,
        help="output of prepare_focus.py — must contain a .done marker",
    )

    # arch / weights
    p.add_argument("--arch", default="vmamba_small", type=str,
                   choices=["vmamba_small"])
    p.add_argument("--vmamba_imagenet_ckpt", default="", type=str,
                   help="ImageNet VMamba init .pth (loaded inside backbone __init__)")
    p.add_argument("--pretrained_weights", default="", type=str,
                   help="OpenUS .pth checkpoint (loaded on top of vmamba init)")
    p.add_argument("--checkpoint_key", default="teacher", type=str,
                   choices=["teacher", "student"])

    # loss
    p.add_argument("--loss_type", default="smooth_l1", type=str,
                   choices=["smooth_l1", "gwd", "kld"],
                   help="rbox regression loss. v1 default smooth_l1. "
                        "gwd/kld require reg_decoded_bbox=True and are "
                        "ablation-only.")

    # CTR
    p.add_argument("--ctr_score_threshold", default=0.3, type=float)
    p.add_argument("--ctr_tolerances", default="0.03,0.05,0.10", type=str,
                   help="comma-separated tolerances for CTR accuracy")

    # optim / schedule
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch_size_per_gpu", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--lr", default=1e-4, type=float)

    # bookkeeping
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--load_from", default="", type=str,
                   help="optional checkpoint path to resume from (mmengine "
                        "treats this as load_from / resume)")
    p.add_argument("--resume", default=False, type=_bool)
    p.add_argument("--dist_url", default="env://", type=str)

    # smoke / debug
    p.add_argument("--smoke_iters", default=0, type=int,
                   help="if > 0: run that many train iters then exit "
                        "(skip val/test). Useful for the 10-iter smoke check.")

    return p


# ---------- override builder ------------------------------------------------

def _build_overrides(args):
    """Build the dict that gets merged into the loaded Config."""
    prepared = Path(args.prepared_dir).resolve()
    trainval = prepared / "trainval"
    test     = prepared / "test"
    test_meta_json = test / "meta.json"

    if not (prepared / ".done").is_file():
        raise SystemExit(
            f"prepared dataset not found ({prepared}/.done missing). Run "
            f"`python -m downstream_tasks.cardiac_detection.prepare_focus "
            f"--data_root {args.data_root} --prepared_dir {prepared}` first."
        )

    tolerances = tuple(float(x) for x in args.ctr_tolerances.split(",") if x.strip())

    cfg_override = dict(
        work_dir=args.output_dir,
        randomness=dict(seed=int(args.seed), deterministic=False),
        train_cfg=dict(max_epochs=int(args.epochs)),
        train_dataloader=dict(
            batch_size=int(args.batch_size_per_gpu),
            num_workers=int(args.num_workers),
            dataset=dict(data_root=str(trainval)),
        ),
        val_dataloader=dict(
            num_workers=int(args.num_workers),
            dataset=dict(data_root=str(test)),
        ),
        test_dataloader=dict(
            num_workers=int(args.num_workers),
            dataset=dict(data_root=str(test)),
        ),
        val_evaluator=dict(
            meta_json_path=str(test_meta_json),
            score_threshold=float(args.ctr_score_threshold),
            tolerances=tolerances,
        ),
        test_evaluator=dict(
            meta_json_path=str(test_meta_json),
            score_threshold=float(args.ctr_score_threshold),
            tolerances=tolerances,
        ),
        optim_wrapper=dict(optimizer=dict(lr=float(args.lr))),
        model=dict(
            backbone=dict(
                vmamba_imagenet_ckpt=args.vmamba_imagenet_ckpt or None,
                openus_ckpt=args.pretrained_weights or None,
                openus_key=args.checkpoint_key,
            ),
        ),
    )
    if args.load_from:
        cfg_override["load_from"] = args.load_from
    if args.resume:
        cfg_override["resume"] = True
    return cfg_override


# ---------- loss-type plumbing ----------------------------------------------

def _apply_loss_override(cfg, loss_type):
    """Override the roi_head bbox loss based on --loss_type."""
    if loss_type == "smooth_l1":
        return
    # GWD / KLD ablation requires reg_decoded_bbox=True on the bbox head.
    bbox_head = cfg.model.roi_head.bbox_head
    bbox_head["reg_decoded_bbox"] = True
    if loss_type == "gwd":
        bbox_head["loss_bbox"] = dict(
            type="GDLoss", loss_type="gwd",
            fun="log1p", tau=1.0, loss_weight=5.0,
        )
    elif loss_type == "kld":
        bbox_head["loss_bbox"] = dict(
            type="KLDLoss", loss_type="kld",
            fun="log1p", tau=1.0, loss_weight=5.0,
        )


# ---------- main ------------------------------------------------------------

def main():
    args = build_argparser().parse_args()

    # Lazy imports: mmrotate is only needed at runtime; the CLI parses and
    # input-validates first so we fail fast on bad args even without mmrotate.
    from mmengine.config import Config, DictAction  # noqa: F401
    from mmengine.runner import Runner
    from mmrotate.utils import register_all_modules
    register_all_modules(init_default_scope=False)

    cfg_path = os.path.join(
        THIS_DIR, "downstream_tasks", "cardiac_detection", "configs",
        "rotated_faster_rcnn_openus_focus.py",
    )
    cfg = Config.fromfile(cfg_path)
    cfg.merge_from_dict(_build_overrides(args))

    _apply_loss_override(cfg, args.loss_type)

    # Smoke-test path: 10 iters, no val/test.
    if args.smoke_iters and args.smoke_iters > 0:
        cfg.train_cfg.max_epochs = 1
        cfg.train_cfg.val_interval = 99999
        cfg.train_dataloader.batch_size = min(
            cfg.train_dataloader.batch_size, 2,
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    runner = Runner.from_cfg(cfg)

    # ---- train --------------------------------------------------------------
    runner.train()

    # ---- final test ---------------------------------------------------------
    if not args.smoke_iters:
        metrics = runner.test()
    else:
        metrics = {}

    # ---- write a single landmark/LVEF-style JSONL record -------------------
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        log_txt = os.path.join(args.output_dir, "log.txt")
        rec = {
            "phase":       "final_test",
            "args":        vars(args),
            "final_test":  {str(k): float(v) if isinstance(v, (int, float)) else v
                            for k, v in metrics.items()},
        }
        with open(log_txt, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"\n[eval_cardiac_detection] final-test metrics:")
        for k, v in metrics.items():
            print(f"  {k:<30s} {v}")
        print(f"\n[eval_cardiac_detection] appended JSONL record to {log_txt}")


if __name__ == "__main__":
    main()
