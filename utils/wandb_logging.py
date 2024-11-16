def create_wandb_config(cfg):
    """Extract relevant experimental parameters from mmseg config."""
    wandb_config = {
        "architecture": {
            "backbone": "swin",
            "neck": "fpn",
            "head": "fpn",
            "input_size": cfg.crop_size,
            "num_classes": cfg.model.decode_head.num_classes,
            "swin_config": {
                "embed_dims": cfg.model.backbone.embed_dims,
                "depths": cfg.model.backbone.depths,
                "num_heads": cfg.model.backbone.num_heads,
                "window_size": cfg.model.backbone.window_size,
            }
        },
        "training": {
            "batch_size": cfg.train_dataloader.batch_size,
            "grad_accumulation": cfg.optim_wrapper.accumulative_counts,
            "max_iters": cfg.train_cfg.max_iters,
            "val_interval": cfg.train_cfg.val_interval,
            "optimizer": {
                "type": cfg.optim_wrapper.optimizer.type,
                "lr": cfg.optim_wrapper.optimizer.lr,
                "weight_decay": cfg.optim_wrapper.optimizer.weight_decay,
            },
            "scheduler": [
                {
                    "type": "LinearLR",
                    "start_factor": cfg.param_scheduler[0].start_factor,
                    "end_factor": cfg.param_scheduler[0].end_factor,
                    "end_iter": cfg.param_scheduler[0].end
                },
                {
                    "type": "PolyLR",
                    "power": cfg.param_scheduler[1].power,
                    "eta_min": cfg.param_scheduler[1].eta_min,
                    "begin_iter": cfg.param_scheduler[1].begin,
                    "end_iter": cfg.param_scheduler[1].end
                }
            ],
            "losses": [loss["type"] for loss in cfg.model.decode_head.loss_decode],
        },
        "augmentation": {
            "resize_ratio_range": cfg.train_pipeline[2].ratio_range,
            "crop_size": cfg.train_pipeline[3].crop_size,
            "flip_prob": cfg.train_pipeline[4].prob,
            "photometric_distortion": True  # presence of this augmentation
        }
    }
    return wandb_config