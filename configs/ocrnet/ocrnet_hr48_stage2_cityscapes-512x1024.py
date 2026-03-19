_base_ = ['./ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py']

# Stage 2 of 3-stage pretraining pipeline: Map → City → RS19
# Fine-tunes the Stage 1 Mapillary checkpoint on Cityscapes (19 classes).
# Heads are re-initialised due to 66→19 class mismatch.
# Best checkpoint feeds into Stage 3 (RS19 fine-tune).

# Set this to the best checkpoint from Stage 1:
#   work_dirs/ocrnet_hr48_stage1_mapillary-512x1024/<timestamp>/best_mIoU_iter_<N>.pth
load_from = None  # <-- SET THIS after Stage 1 completes

# SyncBN for multi-GPU training (3× RTX 6000, matches paper)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=None,  # load_from handles weights; skip redundant ImageNet backbone load
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])

# Fine-tuning LR; warmup from low LR then poly decay
# Effective batch 48 = batch 2/GPU × 3 GPUs × accum 8
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper', optimizer=optimizer, clip_grad=None, accumulative_counts=8)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=1500, end=120000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Stop early if mIoU doesn't improve by >0.2 for 5 consecutive validations (20k iters)
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        min_delta=0.2,
        patience=5)]
