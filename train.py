import os
import logging
from pathlib import Path
import torch
import random
import numpy as np
from argparse import ArgumentParser
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmseg.utils import register_all_modules
from utils.class_weights import setup_weights

def parse_args():
    parser = ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument('--resume-from', help='resume from checkpoint')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--no-validate', action='store_true', help='disable validation')
    return parser.parse_args()

def verify_dataset(cfg):
    """Verify dataset paths exist."""
    data_root = Path(cfg.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}\n"
            f"Please make sure you've downloaded and extracted the ADE20K dataset"
        )
    
    # Check train/val directories
    for split in ['train', 'val']:
        loader_cfg = getattr(cfg, f'{split}_dataloader')
        if not loader_cfg:
            raise ValueError(f"No {split}_dataloader configuration found")
            
        img_path = loader_cfg.dataset.data_prefix.img_path
        ann_path = loader_cfg.dataset.data_prefix.seg_map_path
        
        img_dir = data_root / img_path
        ann_dir = data_root / ann_path
        
        if not img_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(
                f"{split.title()} directories not found:\n"
                f"Images ({img_path}): {img_dir}\n"
                f"Annotations ({ann_path}): {ann_dir}\n"
                f"Please verify your data_root and folder structure"
            )

def setup_environment(cfg, args):
    """Set up training environment and configuration."""
    # Set up work directory
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        cfg.work_dir = os.path.join('./work_dirs',
                                  os.path.splitext(os.path.basename(args.config))[0])
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up weights
    weights_path = os.path.join(cfg.work_dir, 'class_weights.npy')
    cfg = setup_weights(cfg, weights_path)
    
    # Handle resume/validate settings
    if args.resume_from:
        cfg.resume = True
        cfg.load_from = args.resume_from
    
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
    
    # Set random seed if specified
    if args.seed is not None:
        print_log(f'Setting random seed to {args.seed}', 'current')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    return cfg

def main():
    args = parse_args()
    register_all_modules()
    
    # Load and validate config
    cfg = Config.fromfile(args.config)
    verify_dataset(cfg)
    cfg = setup_environment(cfg, args)
    
    print_log(f'Config:\n{cfg.pretty_text}', 'current')
    
    try:
        # Build runner and start training
        runner = Runner.from_cfg(cfg)
        print_log("Starting training...", "current")
        runner.train()
        print_log("Training completed successfully", "current")
    except Exception as e:
        print_log(f"Training failed: {str(e)}", "current", logging.ERROR)
        raise

if __name__ == '__main__':
    main()