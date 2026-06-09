"""mmrotate config: Rotated Faster R-CNN + OpenUS-VMamba backbone on FOCUS.

Self-contained — no ``_base_`` inheritance — so the config does not depend
on where mmrotate's stock configs are installed on disk. Structure follows
mmrotate 1.0.0rc1's ``rotated-faster-rcnn-le90_r50_fpn_1x_dota.py`` (the
ResNet50 baseline used in the paper); only the backbone, neck channels,
num_classes, pipeline, optimizer, and evaluator are swapped.

Note on "rotated Faster R-CNN" architecture: this is *horizontal* RPN +
*rotated* R-CNN head. The RPN proposes horizontal boxes, RoIAlign pools
horizontal RoIs, and the bbox head regresses rbox parameters (cx, cy, w,
h, theta) — the ``predict_box_type='rbox'`` flag is what makes the head
output rotated boxes.

Overridable at runtime by ``eval_cardiac_detection.py``:
    work_dir, randomness.seed
    train_cfg.max_epochs, train_dataloader.batch_size
    optim_wrapper.optimizer.lr
    model.backbone.openus_ckpt, model.backbone.vmamba_imagenet_ckpt
    {train,val,test}_dataloader.dataset.data_root
    {val,test}_evaluator.meta_json_path
"""

# ----------------------------------------------------------------------------
# Constants — runner can override.
# ----------------------------------------------------------------------------
DATA_ROOT_TRAINVAL = ""
DATA_ROOT_TEST     = ""
IMG_SIZE           = 224
NUM_CLASSES        = 2          # thorax, cardiac
angle_version      = "le90"     # mmrotate long-edge 90° convention

# ----------------------------------------------------------------------------
# Custom imports — registers OpenUSVMamba, FOCUSDataset, FocusCTRMetric.
# ----------------------------------------------------------------------------
custom_imports = dict(
    imports=[
        "downstream_tasks.cardiac_detection.backbone_mmrotate",
        "downstream_tasks.cardiac_detection.dataset_focus",
        "downstream_tasks.cardiac_detection.metrics_focus",
    ],
    allow_failed_imports=False,
)

# ----------------------------------------------------------------------------
# Data preprocessor — ImageNet normalisation; grayscale-replicated input.
# ----------------------------------------------------------------------------
data_preprocessor = dict(
    type="mmdet.DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_size_divisor=32,
    boxtype2tensor=False,
)

# ----------------------------------------------------------------------------
# Model — stock Rotated Faster R-CNN schema with OpenUS-VMamba backbone.
# ----------------------------------------------------------------------------
model = dict(
    type="mmdet.FasterRCNN",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="OpenUSVMamba",
        arch="vmamba_small",
        out_indices=(0, 1, 2, 3),
        vmamba_imagenet_ckpt=None,   # filled by runner
        openus_ckpt=None,             # filled by runner
        openus_key="teacher",
        frozen_stages=-1,
        # Full encoder freeze for the OpenUS-vs-EchoCare A/B (head-only
        # fine-tuning). EchoCare config sets the same flag — only the FPN /
        # RPN / R-CNN head train, so any AP/CTR delta is attributable to
        # the encoder representation alone.
        freeze_encoder=True,
    ),
    neck=dict(
        type="mmdet.FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        type="mmdet.RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="mmdet.AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True,
        ),
        bbox_coder=dict(
            type="mmrotate.DeltaXYWHHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            use_box_type=True,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="mmdet.SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0,
        ),
    ),
    roi_head=dict(
        type="mmdet.StandardRoIHead",
        bbox_roi_extractor=dict(
            type="mmdet.SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="mmdet.Shared2FCBBoxHead",
            predict_box_type="rbox",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=NUM_CLASSES,
            reg_predictor_cfg=dict(type="mmdet.Linear"),
            cls_predictor_cfg=dict(type="mmdet.Linear"),
            bbox_coder=dict(
                type="mmrotate.DeltaXYWHTHBBoxCoder",
                angle_version=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
            ),
            reg_class_agnostic=True,
            loss_cls=dict(
                type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0,
            ),
            # v1 default: stock Smooth-L1 on encoded deltas (paper-faithful).
            loss_bbox=dict(
                type="mmdet.SmoothL1Loss", beta=1.0, loss_weight=1.0,
            ),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="mmdet.MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type="mmrotate.RBbox2HBboxOverlaps2D"),
            ),
            sampler=dict(
                type="mmdet.RandomSampler",
                num=256, pos_fraction=0.5, neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="mmdet.MaxIoUAssigner",
                pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                match_low_quality=False, ignore_iof_thr=-1,
                iou_calculator=dict(type="mmrotate.RBbox2HBboxOverlaps2D"),
            ),
            sampler=dict(
                type="mmdet.RandomSampler",
                num=512, pos_fraction=0.25, neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000, max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type="nms_rotated", iou_threshold=0.1),
            max_per_img=100,
        ),
    ),
)

# ----------------------------------------------------------------------------
# Pipelines — 224x224 canvas, keep-ratio resize + letterbox pad.
# ----------------------------------------------------------------------------
train_pipeline = [
    dict(type="mmdet.LoadImageFromFile", color_type="color"),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize",
         scale=(IMG_SIZE, IMG_SIZE), keep_ratio=True),
    dict(type="mmdet.Pad",
         size=(IMG_SIZE, IMG_SIZE), pad_val=dict(img=(0, 0, 0))),
    dict(type="mmdet.RandomFlip",
         prob=0.5, direction=["horizontal", "vertical"]),
    dict(type="mmdet.PackDetInputs"),
]

test_pipeline = [
    dict(type="mmdet.LoadImageFromFile", color_type="color"),
    dict(type="mmdet.Resize",
         scale=(IMG_SIZE, IMG_SIZE), keep_ratio=True),
    dict(type="mmdet.Pad",
         size=(IMG_SIZE, IMG_SIZE), pad_val=dict(img=(0, 0, 0))),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.PackDetInputs",
         meta_keys=(
             "img_id", "img_path", "ori_shape", "img_shape",
             "scale_factor", "pad_shape", "flip", "flip_direction",
         )),
]

# ----------------------------------------------------------------------------
# Dataloaders
# ----------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="FOCUSDataset",
        data_root=DATA_ROOT_TRAINVAL,
        # ann_file = annfiles directory (the DOTADataset.load_data_list
        # ``ann_file != ""`` path walks *.txt files there and pairs each
        # with a same-stem .png under data_prefix['img_path']).
        ann_file="annfiles",
        data_prefix=dict(img_path="images"),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="FOCUSDataset",
        data_root=DATA_ROOT_TEST,
        ann_file="annfiles",
        data_prefix=dict(img_path="images"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# ----------------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------------
val_evaluator = dict(
    type="FocusCTRMetric",
    meta_json_path="",  # set by runner: <prepared_dir>/test/meta.json
    score_threshold=0.3,
    tolerances=(0.03, 0.05, 0.10),
    iou_thrs=[0.5],
)
test_evaluator = val_evaluator

# ----------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=10)
val_cfg   = dict(type="ValLoop")
test_cfg  = dict(type="TestLoop")

# ----------------------------------------------------------------------------
# Optimizer — AdamW, head-only training (encoder is frozen above).
# No backbone paramwise_cfg needed: with freeze_encoder=True, every backbone
# parameter has requires_grad=False, so mmengine excludes them from the
# optimizer automatically.
# ----------------------------------------------------------------------------
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR",
         start_factor=1.0 / 1000.0,
         by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR",
         T_max=100, eta_min=1e-7,
         by_epoch=True, begin=0, end=100,
         convert_to_iter_based=True),
]

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
default_scope = "mmrotate"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        # save_best uses the actual key emitted by FocusCTRMetric (which
        # inherits its top-level keys from DOTAMetric). With iou_thrs=[0.5]
        # mAP == AP50 == mean of per-class AP at IoU=0.5.
        type="CheckpointHook", interval=20,
        save_best="focus/mAP", rule="greater", max_keep_ckpts=2,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="mmdet.DetVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="mmdet.DetLocalVisualizer",
    vis_backends=vis_backends, name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level   = "INFO"
load_from   = None
resume      = False
randomness  = dict(seed=42, deterministic=False)
