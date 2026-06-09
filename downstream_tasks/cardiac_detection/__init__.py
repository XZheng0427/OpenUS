"""US-DINO downstream task 4: rotated-bbox detection on FOCUS (fetal cardiac).

Package layout:
    prepare_focus.py       - one-shot merge of FOCUS training/+validation/ into
                              a 250-image trainval split + 50-image test split,
                              with per-image meta.json and a `.done` marker.
    ctr_utils.py           - pure-numpy CTR + rotated-box inverse-transform.
    backbone_mmrotate.py   - OpenUSVMamba: Backbone_DINOv2_VSSM_2 registered in
                              mmrotate's MODELS registry, with init_weights()
                              that calls _backbone_init.load_openus_backbone().
    dataset_focus.py       - FOCUSDataset: thin subclass of DOTADataset with
                              FOCUS METAINFO ("thorax", "cardiac").
    metrics_focus.py       - FocusCTRMetric: rotated AP + CTR family
                              (ctr_missing_rate, CTR-MAE(valid), CTR-acc@*).
    configs/rotated_faster_rcnn_openus_focus.py - mmrotate detector config.
    _aggregate_multiseed.py - reads log.txt JSONL across seeds.
    tests/                 - smoke + unit tests (numpy + torch only; do not
                              require mmrotate).

The companion top-level runner is `eval_cardiac_detection.py`.
"""
