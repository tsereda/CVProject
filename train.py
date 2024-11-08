import argparse
import os
import time
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules
from utils.class_weights import generate_class_weights, load_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
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
        '--calculate-weights',
        action='store_true',
        help='calculate class weights before training')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Register all modules
    register_all_modules()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Update config based on args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                  os.path.splitext(os.path.basename(args.config))[0])
    
    # Create work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Calculate or load class weights
    weights_path = os.path.join(cfg.work_dir, 'class_weights.npy')
    if args.calculate_weights or not os.path.exists(weights_path):
        print("Calculating class weights...")
        weights = generate_class_weights(
            data_root=cfg.data_root,
            save_path=weights_path,
            method='median'
        )
        print(f"Generated weights shape: {weights.shape}")
        print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    else:
        print("Loading existing class weights...")
        weights = load_class_weights(weights_path)
        print(f"Loaded weights shape: {weights.shape}")
        print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Convert numpy array to list and update config
    weights_list = weights.tolist()
    cfg.model.decode_head.loss_decode = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0
    )
    
    # Set gpu number
    cfg.gpu_ids = range(args.gpus)
    
    # Build runner
    runner = Runner.from_cfg(cfg)
    
    # Start training
    runner.train()

if __name__ == '__main__':
    main()