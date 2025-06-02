# configs/ann/ann_r50-d8_4xb2-40k_railsem19-768x1024.py
_base_ = [
    "../_base_/models/ann_r50-d8.py",
    "../_base_/datasets/railsem19.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

# –– Reduce crop size to lower memory per image ––
crop_size = (768, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(data_preprocessor=data_preprocessor)
model.update(
    decode_head=dict(
        num_classes=19,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    )
)

# –– Keep batch_size=1 but accumulate 2 steps to emulate batch_size=2 ––
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
)

# –– Enable FP16 mixed precision ––
fp16 = dict(loss_scale=512)

# –– Gradient accumulation config ––
optimizer_config = dict(
    type="GradientCumulativeOptimizer",
    cumulative_iters=2,  # two iterations of batch_size=1 = effective batch_size=2
)

# When using gradient accumulation, adjust LR
optimizer = dict(
    type="AdamW",
    lr=3e-05,  # Half of original 6e-05 since effective batch size doubled
    weight_decay=0.01,
)

# Add warmup for stability
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type="PolyLR",
        eta_min=0,
        power=0.9,
        begin=1000,
        end=40000,
        by_epoch=False,
    ),
]
