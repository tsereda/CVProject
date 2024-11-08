# configs/fpn_mobilenetv2_ade20k.py

_base_ = './base_fpn_ade20k.py'

# Model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        strides=(1, 2, 2, 2, 1, 2, 1),
        dilations=(1, 1, 1, 1, 1, 1, 1),
        out_indices=(1, 2, 4, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='mmcls://mobilenet_v2')),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=128,  # Changed to match decode_head input channels
        num_outs=4,
        add_extra_convs='on_lateral'),
    decode_head=dict(
        type='FPNHead',
        in_channels=[128, 128, 128, 128],  # Matches neck out_channels
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,  # Increased to match input channels
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=None)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Optimizer - using AdamW with warmup
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Change to AMP optimizer
    optimizer=dict(
        type='AdamW',
        lr=0.0005,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=0.5)
        }),
    accumulative_counts=2)  # Gradient accumulation to reduce memory usage

# Learning rate scheduler with warmup
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False)
]

# Lighter data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),  # Reduced from 1024
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PackSegInputs')
]

# Dataloader settings
train_dataloader = dict(
    batch_size=8,  # Increased from 8
    num_workers=4,  # Increased from 2
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ADE20KDataset',
        data_root='ADEChallengeData2016',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        pipeline=train_pipeline))

# More frequent validation
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=4000)  # More frequent validation

# Enable mixed precision training
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Add AMP config
amp_cfg = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    loss_scale_init=512.,
    loss_scale_window=1000,
    calibrate=True,  # Add calibration for better stability
    custom_fp16_rules=None)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Add empty cache frequency
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True)
]