import argparse
import os
import time
import logging
import numpy as np
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules
from utils.class_weights import generate_class_weights, load_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir',
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', 
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='random seed')
    parser.add_argument(
        '--calculate-weights',
        action='store_true',
        help='calculate class weights before training')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    return parser.parse_args()

def setup_workdir(cfg, args):
    """Setup working directory."""
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                  os.path.splitext(os.path.basename(args.config))[0])
    
    # Create work_dir
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    
    return cfg

def setup_weights(cfg, weights_path, calculate_weights=False):
    """Calculate or load class weights and update config."""
    try:
        if calculate_weights or not os.path.exists(weights_path):
            print_log("Calculating class weights...", "current")
            weights, stats = generate_class_weights(
                data_root=cfg.data_root,
                save_path=weights_path,
                method='median_frequency_balanced'
            )
            
            # Log statistics from calculation
            print_log(
                f"Weight statistics - Min: {weights.min():.3f}, "
                f"Max: {weights.max():.3f}, Mean: {weights.mean():.3f}", 
                "current"
            )
            
        else:
            print_log("Loading existing class weights...", "current")
            try:
                weights = load_class_weights(weights_path)
            except Exception as e:
                print_log(f"Failed to load weights: {str(e)}", "current", logging.WARNING)
                weights = None
        
        # Ensure weights are valid if present
        if weights is not None:
            if not isinstance(weights, np.ndarray):
                print_log("Invalid weights type - defaulting to no weights", "current", logging.WARNING)
                weights = None
            elif np.any(~np.isfinite(weights)):
                print_log("Non-finite weights detected - defaulting to no weights", "current", logging.WARNING)
                weights = None
            elif len(weights) != 151:  # 150 classes + ignore label
                print_log(f"Incorrect weights shape {len(weights)} - defaulting to no weights", "current", logging.WARNING)
                weights = None
            else:
                # Clip weights to prevent extreme values
                weights = np.clip(weights, 0.1, 10.0)
        
        if weights is not None:
            if 'decode_head' in cfg.model:
                cfg.model.decode_head.loss_decode.class_weight = weights.tolist()
                print_log("Successfully updated config with class weights", "current")
            else:
                print_log("Warning: Could not find decode_head in model config", "current", logging.WARNING)
        else:
            # If no valid weights, ensure class_weight is None in config
            if 'decode_head' in cfg.model and hasattr(cfg.model.decode_head.loss_decode, 'class_weight'):
                cfg.model.decode_head.loss_decode.class_weight = None
                print_log("No valid weights available - disabled class weighting", "current")
            
    except Exception as e:
        print_log(f"Error in weight calculation: {str(e)}", "current", logging.ERROR)
        print_log("Continuing without class weights...", "current", logging.WARNING)
        # Ensure class_weight is None in config
        if 'decode_head' in cfg.model and hasattr(cfg.model.decode_head.loss_decode, 'class_weight'):
            cfg.model.decode_head.loss_decode.class_weight = None
    
    return cfg

def verify_dataset(cfg):
    """Verify dataset configuration and existence."""
    try:
        data_root = Path(cfg.data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset root {data_root} does not exist")
        
        # Check training data
        train_img_dir = data_root / cfg.train_dataloader.dataset.data_prefix.img_path
        train_ann_dir = data_root / cfg.train_dataloader.dataset.data_prefix.seg_map_path
        if not train_img_dir.exists() or not train_ann_dir.exists():
            raise FileNotFoundError(
                f"Training directories not found:\n"
                f"Images: {train_img_dir}\n"
                f"Annotations: {train_ann_dir}"
            )
        
        # Check validation data
        val_img_dir = data_root / cfg.val_dataloader.dataset.data_prefix.img_path
        val_ann_dir = data_root / cfg.val_dataloader.dataset.data_prefix.seg_map_path
        if not val_img_dir.exists() or not val_ann_dir.exists():
            raise FileNotFoundError(
                f"Validation directories not found:\n"
                f"Images: {val_img_dir}\n"
                f"Annotations: {val_ann_dir}"
            )
            
    except Exception as e:
        print_log(f"Dataset verification failed: {str(e)}", "current", logging.ERROR)
        raise

def main():
    args = parse_args()
    
    # Register all modules
    register_all_modules()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Setup work directory
    cfg = setup_workdir(cfg, args)
    
    # Verify dataset configuration
    verify_dataset(cfg)
    
    # Setup weights
    weights_path = os.path.join(cfg.work_dir, 'class_weights.npy')
    cfg = setup_weights(cfg, weights_path, args.calculate_weights)
    
    # Update config based on args
    if args.resume_from:
        cfg.resume = True
        cfg.load_from = args.resume_from
    
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
    
    # Set gpu ids
    if args.gpus is not None:
        cfg.gpu_ids = range(args.gpus)
    
    # Set seed if specified
    if args.seed is not None:
        print_log(f'Set random seed to {args.seed}', 'current')
        import torch
        import numpy as np
        import random
        
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Print config
    print_log(f'Config:\n{cfg.pretty_text}', 'current')
    
    try:
        # Build runner
        runner = Runner.from_cfg(cfg)
        
        # Start training
        print_log("Starting training...", "current")
        runner.train()
        print_log("Training completed successfully", "current")
        
    except Exception as e:
        print_log(f"Training failed with error: {str(e)}", "current", logging.ERROR)
        raise

if __name__ == '__main__':
    main()