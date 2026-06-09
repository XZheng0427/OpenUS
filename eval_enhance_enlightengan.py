"""USenhance-adapted EnlightenGAN — training + eval.

"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils

from downstream_tasks.enhance._runner import (
    build_train_hq_reference, compute_final_metrics,
    gather_test_rows, save_results_json,
)
from downstream_tasks.enhance.metrics_enhance import (
    write_png, rank_scoped_dir, merge_rank_dirs,
)

from downstream_tasks.enhance_enlightengan.networks import (
    GANLoss, VGGFeatureLoss, define_D, define_G,
)
from downstream_tasks.enhance_enlightengan.dataset_enlightengan import (
    EnlightenGANUnpairedDataset, EnlightenGANTestDataset,
    unpaired_collate, test_collate, make_worker_init_fn,
)
from downstream_tasks.enhance_enlightengan.transforms_enlightengan import (
    make_train_image_transform, make_eval_image_transform,
)


# ---------- GAN loss helpers (RaGAN global, LSGAN local) --------------------

def ragan_g(gan, d, fake, real):
    pr, pf = d(real), d(fake)
    return 0.5 * (gan(pr - pf.mean(), False) + gan(pf - pr.mean(), True))


def ragan_d(gan, d, fake_detached, real):
    pr, pf = d(real), d(fake_detached)
    return 0.5 * (gan(pr - pf.mean(), True) + gan(pf - pr.mean(), False))


def sample_patch_coords(size, patch, n):
    coords = []
    for _ in range(n):
        h = torch.randint(0, size - patch + 1, (1,)).item()
        w = torch.randint(0, size - patch + 1, (1,)).item()
        coords.append((h, w))
    return coords


# ---------- final test dump (mirrors _runner.dump_test_enhanced) ------------

@torch.no_grad()
def dump_test_enlightengan(G, loader, output_dir, args, subdir="enhanced_test"):
    G.eval()
    rank_dir = rank_scoped_dir(output_dir, subdir)
    rows = []
    for A, A_gray, metas in loader:
        A = A.cuda(non_blocking=True)
        A_gray = A_gray.cuda(non_blocking=True)
        out, _ = G(A, A_gray)
        img01 = (out * 0.5 + 0.5).clamp(0.0, 1.0)
        for i, m in enumerate(metas):
            fname = f"{m['split']}_{m['organ']}_{m['stem']}.png"
            write_png(img01[i], os.path.join(rank_dir, fname))
            rows.append({"fname": fname, "organ": m["organ"],
                         "lq_path": m["lq_path"], "hq_path": m["hq_path"]})
    return rows


@torch.no_grad()
def dump_holdout_enlightengan(G, loader, output_dir, args, subdir="holdout_enhanced"):
    G.eval()
    rank_dir = rank_scoped_dir(output_dir, subdir)
    for A, A_gray, metas in loader:
        A = A.cuda(non_blocking=True)
        A_gray = A_gray.cuda(non_blocking=True)
        out, _ = G(A, A_gray)
        img01 = (out * 0.5 + 0.5).clamp(0.0, 1.0)
        for i, m in enumerate(metas):
            write_png(img01[i], os.path.join(rank_dir, f"holdout_{m['stem']}.png"))


def main(args):
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)
    print("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))

    # ---------- data (shared manifest = same split/seed) ----------
    train_tf = make_train_image_transform(image_size=args.image_size)
    eval_tf  = make_eval_image_transform(image_size=args.image_size)

    train_ds = EnlightenGANUnpairedDataset(args.manifest_path, args.data_root, train_tf)
    test_ds  = EnlightenGANTestDataset(args.manifest_path, args.data_root, eval_tf)
    print(f"data: train(A)={len(train_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=unpaired_collate, worker_init_fn=make_worker_init_fn(args.seed),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=test_collate, drop_last=False,
    )

    # train-split records for the FID reference (same as every pipeline)
    with open(args.manifest_path) as f:
        train_records = [r for r in json.load(f) if r.get("split") == "train"]

    # ---------- generator ----------
    if args.generator == "unet":
        G = define_G(use_norm=args.use_norm, skip=1.0).cuda()
        g_params = list(G.parameters())
    elif args.generator == "vmamba":
        # US-DINO VMamba/OpenUS encoder + EnlightenGAN-style attention decoder.
        from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2
        from downstream_tasks._backbone_init import load_openus_backbone
        from downstream_tasks.enhance_enlightengan.generator_vmamba import build_vmamba_generator
        backbone = Backbone_DINOv2_VSSM_2(pretrained=args.pretrained_vmamba_init)
        if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
            load_openus_backbone(backbone, args.pretrained_weights, key=args.checkpoint_key)
        else:
            print(f"WARNING: no OpenUS weights ({args.pretrained_weights!r}); ImageNet init only")
        G = build_vmamba_generator(backbone, freeze_encoder=args.freeze_encoder, skip=1.0).cuda()
    elif args.generator == "echocare":
        from downstream_tasks.enhance_enlightengan.generator_vmamba import build_echocare_generator
        G = build_echocare_generator(args.pretrained_weights,
                                     freeze_encoder=args.freeze_encoder,
                                     image_size=args.image_size).cuda()
    elif args.generator == "usfm":
        from downstream_tasks.enhance_enlightengan.generator_vmamba import build_usfm_generator
        G = build_usfm_generator(args.pretrained_weights, image_size=args.image_size).cuda()
    elif args.generator == "simmim":
        from downstream_tasks.enhance_enlightengan.generator_vmamba import build_simmim_generator
        G = build_simmim_generator(args.pretrained_weights, image_size=args.image_size).cuda()
    else:
        raise ValueError(f"unknown --generator {args.generator!r}")

    if args.generator == "unet":
        pass  # g_params already set above
    else:
        # hybrid generators expose .decoder / .extractor; opt_g = decoder always
        # + encoder trainable params (only when finetune) at lr*encoder_lr_scale.
        dec_params = list(G.decoder.parameters())
        enc_params = [p for p in G.extractor.parameters() if p.requires_grad]
        g_params = [{"params": dec_params, "lr": args.lr}]
        if enc_params:
            g_params.append({"params": enc_params, "lr": args.lr * args.encoder_lr_scale})
        print(f"{args.generator} generator: freeze_encoder={args.freeze_encoder}  "
              f"decoder={sum(p.numel() for p in dec_params):,}  "
              f"encoder_trainable={sum(p.numel() for p in enc_params):,}")

    D_A = define_D(input_nc=3, ndf=args.ndf, n_layers=args.n_layers_global).cuda()
    D_P = define_D(input_nc=3, ndf=args.ndf, n_layers=args.n_layers_local).cuda()
    gan = GANLoss().cuda()
    vgg = VGGFeatureLoss().cuda()
    print(f"params: G={sum(p.numel() for p in G.parameters()):,}  "
          f"D_A={sum(p.numel() for p in D_A.parameters()):,}  "
          f"D_P={sum(p.numel() for p in D_P.parameters()):,}")

    opt_g = torch.optim.Adam(g_params, lr=args.lr, betas=(args.beta1, 0.999))
    opt_da = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_dp = torch.optim.Adam(D_P.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # linear LR decay: flat for `niter`, then linear -> 0 over `niter_decay`
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - args.niter) / float(args.niter_decay + 1)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda)
              for o in (opt_g, opt_da, opt_dp)]

    n_patches = 1 + args.num_extra_patches
    ps = args.patch_size

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.output_dir, "log.txt")

    # ---------- training ----------
    for epoch in range(args.epochs):
        G.train(); D_A.train(); D_P.train()
        agg = {k: 0.0 for k in ("g", "g_global", "g_local", "vgg_g", "vgg_p", "d_a", "d_p")}
        n = 0
        for A, A_gray, B in train_loader:
            A = A.cuda(non_blocking=True); A_gray = A_gray.cuda(non_blocking=True)
            B = B.cuda(non_blocking=True)
            S = A.shape[-1]
            coords = sample_patch_coords(S, ps, n_patches)

            fake_B, _ = G(A, A_gray)

            # ---- (1) update G ----
            loss_g_global = ragan_g(gan, D_A, fake_B, B)
            loss_g_local = A.new_zeros(())
            loss_vgg_patch = A.new_zeros(())
            for (h, w) in coords:
                fp = fake_B[:, :, h:h + ps, w:w + ps]
                ip = A[:, :, h:h + ps, w:w + ps]
                loss_g_local = loss_g_local + gan(D_P(fp), True)
                loss_vgg_patch = loss_vgg_patch + vgg(fp, ip)
            loss_g_local = loss_g_local / n_patches
            loss_vgg_patch = loss_vgg_patch / n_patches
            loss_vgg_global = vgg(fake_B, A)
            loss_g = (loss_g_global + loss_g_local
                      + args.vgg_weight * (loss_vgg_global + loss_vgg_patch))
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            fake_det = fake_B.detach()

            # ---- (2) update D_A (global RaGAN) ----
            loss_d_a = ragan_d(gan, D_A, fake_det, B)
            opt_da.zero_grad(); loss_d_a.backward(); opt_da.step()

            # ---- (3) update D_P (local LSGAN over patches) ----
            loss_d_p = A.new_zeros(())
            for (h, w) in coords:
                fp = fake_det[:, :, h:h + ps, w:w + ps]
                rp = B[:, :, h:h + ps, w:w + ps]
                loss_d_p = loss_d_p + 0.5 * (gan(D_P(rp), True) + gan(D_P(fp), False))
            loss_d_p = loss_d_p / n_patches
            opt_dp.zero_grad(); loss_d_p.backward(); opt_dp.step()

            agg["g"] += float(loss_g); agg["g_global"] += float(loss_g_global)
            agg["g_local"] += float(loss_g_local); agg["vgg_g"] += float(loss_vgg_global)
            agg["vgg_p"] += float(loss_vgg_patch); agg["d_a"] += float(loss_d_a)
            agg["d_p"] += float(loss_d_p); n += 1

        for s in scheds:
            s.step()
        means = {k: v / max(n, 1) for k, v in agg.items()}
        log = {"epoch": epoch, "lr": opt_g.param_groups[0]["lr"], **means}
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        print(f"epoch {epoch:3d}  lr={log['lr']:.2e}  "
              f"G={means['g']:.4f} (glob={means['g_global']:.4f} loc={means['g_local']:.4f} "
              f"vgg={means['vgg_g']:.4f}/{means['vgg_p']:.4f})  "
              f"D_A={means['d_a']:.4f}  D_P={means['d_p']:.4f}")

    # ---------- save final generator ----------
    torch.save({"epoch": args.epochs, "G": G.state_dict(),
                "D_A": D_A.state_dict(), "D_P": D_P.state_dict(),
                "args": vars(args)},
               os.path.join(args.output_dir, "model_final.pth"))

    # ---------- final test dump + IQA + FID (shared eval) ----------
    rows = dump_test_enlightengan(G, test_loader, args.output_dir, args, "enhanced_test")
    all_rank_rows = gather_test_rows(rows)
    ref_dir = build_train_hq_reference(
        data_root=args.data_root, train_records=train_records,
        output_dir=args.output_dir, subdir="hq_reference", image_size=args.image_size,
    )
    results = compute_final_metrics(
        output_dir=args.output_dir, test_rows_per_rank=all_rank_rows,
        enhanced_subdir="enhanced_test", reference_dir=ref_dir,
        image_size=args.image_size, device=torch.device("cuda"),
    )
    results["pipeline"] = "enlightengan" if args.generator == "unet" else f"enlightengan_{args.generator}"
    # USFM/SimMIM are strict-frozen by their extractor regardless of the flag.
    _frozen = args.freeze_encoder or args.generator in ("usfm", "simmim")
    results["encoder_policy"] = (
        "taskspecific" if args.generator == "unet"
        else ("frozen" if _frozen else "finetune"))
    results["image_size"] = args.image_size
    results["seed"] = args.seed
    save_results_json(args.output_dir, results)
    print("\n========== FINAL TEST METRICS ==========")
    print(f"  pipeline={results['pipeline']}  policy={results['encoder_policy']}  "
          f"img={args.image_size}  seed={args.seed}  n_test={results.get('n_test_images')}")
    print(f"  FID = {results.get('fid'):.4f}")
    for k, v in results.get("iqa", {}).get("overall", {}).items():
        print(f"  {k:<10s} = {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")

    # ---------- optional holdout dump ----------
    if args.holdout_manifest and os.path.isfile(args.holdout_manifest):
        from downstream_tasks.enhance.dataset_enhance import USenhanceHoldoutDataset
        holdout_ds = USenhanceHoldoutDataset(
            args.holdout_manifest, args.data_root, eval_tf)
        # holdout dataset returns (lq, meta); wrap to (A, A_gray, meta) for the dumper
        from downstream_tasks.enhance_enlightengan.transforms_enlightengan import attention_for

        class _Wrap(torch.utils.data.Dataset):
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                lq, m = self.ds[i]
                return lq, attention_for(lq), m
        holdout_loader = DataLoader(
            _Wrap(holdout_ds), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=test_collate)
        dump_holdout_enlightengan(G, holdout_loader, args.output_dir, args)
        merge_rank_dirs(args.output_dir, "holdout_enhanced")


def build_argparser():
    p = argparse.ArgumentParser("USenhance-adapted EnlightenGAN")
    # data
    p.add_argument("--data_root", required=True, type=str)
    p.add_argument("--manifest_path", required=True, type=str)
    p.add_argument("--holdout_manifest", default="", type=str)
    p.add_argument("--image_size", default=256, type=int)
    # generator selection
    p.add_argument("--generator", default="unet", type=str,
                   choices=["unet", "vmamba", "echocare", "usfm", "simmim"],
                   help="'unet' = EnlightenGAN attention U-Net (default); "
                        "else = <encoder> + EnlightenGAN attn decoder inside the "
                        "same RaGAN/local-D/SFP recipe.")
    # vmamba-generator-only flags (ignored when --generator unet)
    p.add_argument("--pretrained_vmamba_init", default="", type=str,
                   help="ImageNet VMamba ckpt for backbone init")
    p.add_argument("--pretrained_weights", default="", type=str,
                   help="OpenUS .pth checkpoint loaded on top")
    p.add_argument("--checkpoint_key", default="teacher", type=str,
                   choices=["teacher", "student"])
    p.add_argument("--freeze_encoder", default=False, type=utils.bool_flag,
                   help="vmamba: freeze the OpenUS encoder (decoder-only training)")
    p.add_argument("--encoder_lr_scale", default=0.1, type=float,
                   help="vmamba finetune: encoder lr = lr * this")
    # model
    p.add_argument("--use_norm", default=True, type=utils.bool_flag)
    p.add_argument("--ndf", default=64, type=int)
    p.add_argument("--n_layers_global", default=5, type=int)
    p.add_argument("--n_layers_local", default=4, type=int)
    p.add_argument("--patch_size", default=32, type=int)
    p.add_argument("--num_extra_patches", default=5, type=int,
                   help="extra random patches; total = 1 + this = 6")
    p.add_argument("--vgg_weight", default=1.0, type=float)
    # optim
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--beta1", default=0.5, type=float)
    p.add_argument("--niter", default=100, type=int, help="flat-LR epochs")
    p.add_argument("--niter_decay", default=100, type=int, help="linear-decay epochs")
    # bookkeeping
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output_dir", required=True, type=str)
    args = p.parse_args()
    args.epochs = args.niter + args.niter_decay
    return args


if __name__ == "__main__":
    main(build_argparser())
