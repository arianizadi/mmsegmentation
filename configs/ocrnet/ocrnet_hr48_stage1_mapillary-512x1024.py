_base_ = [
    '../_base_/models/ocrnet_hr18.py',
    '../_base_/datasets/mapillary_v1.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

# Stage 1 of 3-stage pretraining pipeline: Map → City → RS19
# Starts from ImageNet-pretrained HRNet-W48 backbone only (no Cityscapes bias).
# Heads train from scratch on Mapillary (66 classes).
# Best checkpoint feeds into Stage 2 (Cityscapes fine-tune).

# SyncBN for multi-GPU training (3× RTX 6000, matches paper)
norm_cfg = dict(type='SyncBN', requires_grad=True)

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        norm_cfg=norm_cfg,
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
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
            num_classes=66,
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
            num_classes=66,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# LR=0.01 matching paper (effective batch 48 = batch 2/GPU × 3 GPUs × accum 8)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None, accumulative_counts=8)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=1500, end=500000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=500000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        save_best='mIoU',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))

# Stop early if mIoU doesn't improve by >0.2 for 5 consecutive validations (40k iters)
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        min_delta=0.2,
        patience=5)]
