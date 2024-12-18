dataset_type = 'ADE20KDataset'
data_root = 'ADEChallengeData2016/'

crop_size = (512, 512)
img_scale = (1024, 512)

model = dict(
   type='EncoderDecoder',
   data_preprocessor=dict(
       type='SegDataPreProcessor',
       mean=[123.675, 116.28, 103.53],
       std=[58.395, 57.12, 57.375],
       bgr_to_rgb=True,
       pad_val=0,
       seg_pad_val=255,
       size_divisor=32),
   backbone=dict(
       type='SwinTransformer',
       pretrain_img_size=384,
       embed_dims=128,
       depths=[2, 2, 18, 2],
       num_heads=[4, 8, 16, 32],
       window_size=12,
       use_abs_pos_embed=False,
       drop_path_rate=0.2,
       patch_norm=True,
       patch_size=4,
       mlp_ratio=4,
       strides=(4, 2, 2, 2),
       out_indices=(0, 1, 2, 3),
       qkv_bias=True,
       qk_scale=None,
       drop_rate=0.,
       attn_drop_rate=0.,
       with_cp=True,
       init_cfg=dict(
           type='Pretrained',
           checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth')),
   neck=dict(
       type='FPN',
       in_channels=[128, 256, 512, 1024],
       out_channels=256,
       num_outs=4),
   decode_head=dict(
       type='FPNHead',
       in_channels=[256, 256, 256, 256],
       in_index=[0, 1, 2, 3],
       feature_strides=[4, 8, 16, 32],
       channels=256,
       dropout_ratio=0.1,
       num_classes=151,
       norm_cfg=dict(type='BN', requires_grad=True),
       align_corners=False,
       loss_decode=[
           dict(
               type='CrossEntropyLoss',
               use_sigmoid=False,
               loss_weight=1.,
               class_weight=None,
               avg_non_ignore=True
           ),
           dict(
               type='DiceLoss',
               use_sigmoid=False,
               loss_weight=0.4
           )
       ]),
   train_cfg=dict(),
   test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomRotate',
        prob=0.3,
        degree=(-10, 10),
        pad_val=0,
        seg_pad_val=255),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='Pad', size=crop_size),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PackSegInputs')
]

# Modified validation pipeline to ensure consistent sizes
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=crop_size,
        keep_ratio=False),
    dict(type='Pad', size=crop_size),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    ignore_label=255)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=dict(max_norm=6.0, norm_type=2),
    accumulative_counts=2)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        end_factor=1,  # Added to ensure clear decrease
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=2000,
        end=40000,
        by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=2500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend')
]


visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    save_dir='visual_results'
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False