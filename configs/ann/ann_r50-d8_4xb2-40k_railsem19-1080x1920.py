# configs/ann/ann_r50-d8_4xb2-40k_railsem19-768x1024.py
_base_ = [
    "../_base_/models/ann_r50-d8.py",
    "../_base_/datasets/railsem19.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

#–– Reduce crop size to lower memory per image ––
crop_size = (768, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(data_preprocessor=data_preprocessor)
model.update(
    decode_head=dict(
        num_classes=19,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    )
)

#–– Keep batch_size=1 but accumulate 2 steps to emulate batch_size=2 ––
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
)

#–– Enable FP16 mixed precision ––
fp16 = dict(loss_scale=512)

#–– Gradient accumulation config ––
optimizer_config = dict(
    type="GradientCumulativeOptimizer",
    cumulative_iters=2,  # two iterations of batch_size=1 = effective batch_size=2
)

# (Optional) If you need to lower backbone LR when using accumulation:
# optimizer = dict(
#     type="AdamW",
#     lr=6e-05,        # halve original LR if you doubled effective BS
#     weight_decay=0.01,
# )

