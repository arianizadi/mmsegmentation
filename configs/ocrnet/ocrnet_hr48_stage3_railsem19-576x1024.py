_base_ = ['./ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py']

# Stage 3 of 3-stage pretraining pipeline: Map → City → RS19
# Fine-tunes the Stage 2 Cityscapes checkpoint on RailSem19 (19 classes).
# Both datasets share 19 classes so all weights load cleanly.

# Set this to the best checkpoint from Stage 2:
#   work_dirs/ocrnet_hr48_stage2_cityscapes-512x1024/<timestamp>/best_mIoU_iter_<N>.pth
load_from = 'work_dirs/ocrnet_hr48_stage2_cityscapes-512x1024/best_mIoU_iter_88000.pth'

dataset_type = 'RailSem19Dataset'
data_root = 'data/RailSem19/'
crop_size = (576, 1024)

# SyncBN for multi-GPU training (3× RTX 6000, matches paper)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(size=crop_size)
model = dict(
    pretrained=None,  # load_from handles weights; skip redundant ImageNet backbone load
    data_preprocessor=dict(size=crop_size),
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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[(288, 512), (576, 1024), (1152, 2048)],
        resize_type='Resize',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='GaussianBlur', sigma_range=(0.15, 1.3), prob=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.3,
                contrast_limit=0.4,
                p=0.5)
        ],
        keymap=dict(img='image', gt_seg_map='mask')),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(576, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/annotations'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/annotations'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/annotations'),
        pipeline=test_pipeline))

# Fine-tuning LR on small dataset; warmup then poly decay
# Effective batch 48 = batch 2/GPU × 3 GPUs × accum 8
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper', optimizer=optimizer, clip_grad=None, accumulative_counts=8)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=1500, end=500000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=500000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        save_best='mIoU',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# Stop early if mIoU doesn't improve by >0.5 for 5 consecutive validations (20k iters)
# RS19 is small so gains tend to be larger — use higher min_delta
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        min_delta=0.5,
        patience=5)]

