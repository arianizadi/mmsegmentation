_base_ = ['./bisenetv2_fcn_4xb8-160k_cityscapes-1024x1024.py']
dataset_type = 'RailSem19Dataset'

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_root = 'data/RailSem19/'
crop_size = (576, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=19),
    auxiliary_head=[
        dict(type='FCNHead', in_channels=16, channels=16, num_convs=2, num_classes=19, in_index=1,
             norm_cfg=norm_cfg, concat_input=False, align_corners=False,
             loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(type='FCNHead', in_channels=32, channels=64, num_convs=2, num_classes=19, in_index=2,
             norm_cfg=norm_cfg, concat_input=False, align_corners=False,
             loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(type='FCNHead', in_channels=64, channels=256, num_convs=2, num_classes=19, in_index=3,
             norm_cfg=norm_cfg, concat_input=False, align_corners=False,
             loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(type='FCNHead', in_channels=128, channels=1024, num_convs=2, num_classes=19, in_index=4,
             norm_cfg=norm_cfg, concat_input=False, align_corners=False,
             loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
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
                brightness_limit=0.9,
                contrast_limit=0.0,
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
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/annotations'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images', seg_map_path='val/annotations'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/annotations'),
        pipeline=test_pipeline))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth'

optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None, accumulative_counts=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=2.0,
        begin=0,
        end=160000,
        by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False))
