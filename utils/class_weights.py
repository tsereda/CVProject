import os
import numpy as np
import logging
from pathlib import Path

def load_class_ratios_from_file(ratio_file_path):
    """
    Load class ratios from the ADE20K objectInfo150.txt file.
    
    Args:
        ratio_file_path (str): Path to objectInfo150.txt
    
    Returns:
        np.ndarray: Array of class ratios including ignore label (151 classes total)
    """
    try:
        # Initialize array for 150 classes + ignore label
        ratios = np.zeros(151, dtype=np.float32)
        
        with open(ratio_file_path, 'r') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1  # Convert to 0-based index
                    ratio = float(parts[1])
                    ratios[idx] = ratio
        
        # Validate ratios
        if np.any(~np.isfinite(ratios[:-1])):  # Exclude ignore label
            raise ValueError("Non-finite values found in class ratios")
        if not np.allclose(ratios[:-1].sum(), 1.0, rtol=1e-3):
            logging.warning("Class ratios don't sum to 1.0 (sum: %.4f)", ratios[:-1].sum())
        
        return ratios
        
    except Exception as e:
        raise ValueError(f"Error loading class ratios: {str(e)}")

def compute_class_weights(ratios, method='median', beta=0.9999):
    """
    Compute class weights from ratios using different balancing methods.
    
    Args:
        ratios (np.ndarray): Class ratios
        method (str): 'median' or 'inverse'
        beta (float): Trimming parameter for inverse method
    
    Returns:
        np.ndarray: Class weights including ignore label
    """
    if method == 'median':
        # Median frequency balancing
        median_freq = np.median(ratios[ratios > 0])
        weights = np.where(ratios > 0, median_freq / ratios, 1.0)
    
    elif method == 'inverse':
        # Inverse frequency with beta-trimming
        weights = np.where(ratios > 0,
                          (1 - beta) / (1 - beta ** ratios),
                          1.0)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Clip extreme values
    weights = np.clip(weights, 0.1, 10.0)
    
    # Normalize weights to have mean 1
    weights = weights / weights.mean()
    
    # Set ignore label weight to 0
    weights[-1] = 0
    
    return weights

def setup_weights(cfg, weights_path=None, method='median'):
    """
    Setup class weights for training, either loading from objectInfo150.txt
    or from a pre-computed weights file.
    
    Args:
        cfg: Config object containing data_root
        weights_path (str, optional): Path to save/load pre-computed weights
        method (str): Weighting method ('median' or 'inverse')
    
    Returns:
        cfg: Updated config with weights
    """
    try:
        weights = None
        
        # Try to load pre-computed weights if path provided
        if weights_path and os.path.exists(weights_path):
            try:
                weights = np.load(weights_path)
                logging.info("Loaded pre-computed weights from %s", weights_path)
            except Exception as e:
                logging.warning("Failed to load weights: %s", str(e))
        
        # If no weights loaded, compute from objectInfo150.txt
        if weights is None:
            ratio_file = os.path.join(cfg.data_root, 'objectInfo150.txt')
            if os.path.exists(ratio_file):
                ratios = load_class_ratios_from_file(ratio_file)
                weights = compute_class_weights(ratios, method=method)
                logging.info("Computed weights from class ratios")
                
                # Save weights if path provided
                if weights_path:
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    np.save(weights_path, weights)
                    logging.info("Saved weights to %s", weights_path)
            else:
                logging.warning("objectInfo150.txt not found at %s", ratio_file)
                return cfg
        
        # In setup_weights function, change the config update to:
        if weights is not None and len(weights) == 151:
            if 'decode_head' in cfg.model:
                if isinstance(cfg.model.decode_head.loss_decode, list):
                    # Find CrossEntropyLoss in the list
                    for loss in cfg.model.decode_head.loss_decode:
                        if loss['type'] == 'CrossEntropyLoss':
                            loss['class_weight'] = weights.tolist()
                            logging.info("Updated CrossEntropyLoss with class weights")
                            break
                else:
                    # Single loss case
                    cfg.model.decode_head.loss_decode.class_weight = weights.tolist()
                logging.info("Updated config with class weights")
            
    except Exception as e:
        logging.error("Error setting up weights: %s", str(e))
        # Ensure no invalid weights in config
        if 'decode_head' in cfg.model and hasattr(cfg.model.decode_head.loss_decode, 'class_weight'):
            cfg.model.decode_head.loss_decode.class_weight = None
    
    return cfg