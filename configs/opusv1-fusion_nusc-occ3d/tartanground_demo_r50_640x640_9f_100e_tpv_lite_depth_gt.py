_base_ = ['./tartanground_demo_r50_640x640_9f_100e_tpv_lite_depth.py']

# GT-depth ablation:
# Keep TPV architecture/training setup unchanged and only switch
# depth pseudo-point source from mapanything depth to GT-depth pkl files.
train_ann_file = f'{_base_.dataset_root}train_with_depth.pkl'
val_ann_file = f'{_base_.dataset_root}val_with_depth.pkl'
test_ann_file = f'{_base_.dataset_root}test_with_depth.pkl'

train_dataloader = dict(dataset=dict(ann_file=train_ann_file))
val_dataloader = dict(dataset=dict(ann_file=val_ann_file))
test_dataloader = dict(dataset=dict(ann_file=test_ann_file))
