_base_ = [
    "../_base_/models/ann_r50-d8.py",
    "../_base_/datasets/railsem19.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
crop_size = (1080, 1920)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

# Modify the model to match RailSem19 classes
model.update(
    decode_head=dict(
        num_classes=19,  # Number of classes in RailSem19
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    )
)
