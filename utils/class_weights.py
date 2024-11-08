import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import mmcv

def calculate_class_frequencies(data_root, num_classes=150):
    """
    Calculate class frequencies from ADE20K training annotations.
    
    Args:
        data_root (str): Path to ADE20K dataset root (e.g., 'ADEChallengeData2016/')
        num_classes (int): Number of classes (default: 150 for ADE20K)
    
    Returns:
        np.ndarray: Array of class frequencies normalized by total pixels
    """
    annotation_dir = os.path.join(data_root, 'annotations/training')
    class_pixels = np.zeros(num_classes, dtype=np.int64)
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.png')]
    
    print("Calculating class frequencies...")
    for filename in tqdm(annotation_files):
        filepath = os.path.join(annotation_dir, filename)
        # Load annotation image
        ann = np.array(Image.open(filepath))
        # Count pixels per class
        for class_id in range(num_classes):
            class_pixels[class_id] += np.sum(ann == class_id)
    
    # Convert to frequencies
    total_pixels = class_pixels.sum()
    class_frequencies = class_pixels / total_pixels
    
    return class_frequencies

def compute_class_weights(frequencies, method='median', beta=0.9999):
    """
    Compute class weights using different balancing methods.
    
    Args:
        frequencies (np.ndarray): Class frequencies
        method (str): Weighting method ('median' or 'inverse')
        beta (float): Trimming parameter for inverse frequency to avoid extreme weights
    
    Returns:
        np.ndarray: Class weights
    """
    if method == 'median':
        # Median frequency balancing
        median_freq = np.median(frequencies)
        weights = median_freq / frequencies
        # Clip extreme values
        weights = np.clip(weights, 0.1, 10.0)
        
    elif method == 'inverse':
        # Inverse frequency with beta-trimming
        weights = (1 - beta) / (1 - beta ** frequencies)
        # Clip extreme values
        weights = np.clip(weights, 0.1, 10.0)
        
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights so their mean is close to 1
    weights = weights / weights.mean()
    
    return weights

def generate_class_weights(data_root, save_path=None, method='median', beta=0.9999):
    """
    Generate and optionally save class weights for ADE20K dataset.
    
    Args:
        data_root (str): Path to ADE20K dataset root
        save_path (str, optional): Path to save weights. If None, weights are not saved
        method (str): Weighting method ('median' or 'inverse')
        beta (float): Trimming parameter for inverse frequency method
    
    Returns:
        np.ndarray: Computed class weights
        str: Path to saved weights if save_path is provided
    """
    # Calculate class frequencies
    frequencies = calculate_class_frequencies(data_root)
    
    # Compute weights
    weights = compute_class_weights(frequencies, method=method, beta=beta)
    
    # Save weights if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, weights)
        print(f"Saved class weights to {save_path}")
    
    return weights

def load_class_weights(weights_path):
    """
    Load pre-computed class weights.
    
    Args:
        weights_path (str): Path to the .npy file containing class weights
    
    Returns:
        np.ndarray: Class weights
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Class weights file not found: {weights_path}")
    return np.load(weights_path)