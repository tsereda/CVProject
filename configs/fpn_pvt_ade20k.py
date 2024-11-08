# PVT-Large configuration
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PVT',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_large_v2.pth'),
        pretrained=None),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))