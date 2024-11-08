#!/usr/bin/env python3
# utils/class_weights.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

def calculate_class_frequencies(data_root, num_classes=150):
    """
    Calculate class frequencies from ADE20K training annotations with enhanced debugging.
    
    Args:
        data_root (str): Path to ADE20K dataset root
        num_classes (int): Number of classes (default: 150 for ADE20K)
    
    Returns:
        np.ndarray: Array of class frequencies normalized by total pixels
        dict: Statistics about the dataset
    """
    annotation_dir = os.path.join(data_root, 'annotations/training')
    class_pixels = np.zeros(num_classes, dtype=np.int64)
    stats = {
        'total_images': 0,
        'invalid_labels': 0,
        'max_label_found': -1,
        'problematic_files': [],
        'label_histogram': np.zeros(256, dtype=np.int64),  # Track all possible label values
    }
    
    # Verify directory exists
    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.png')]
    if not annotation_files:
        raise ValueError(f"No PNG annotation files found in {annotation_dir}")
    
    stats['total_images'] = len(annotation_files)
    
    print("\n=== Starting Label Analysis ===")
    print(f"Processing {len(annotation_files)} annotation files...")
    
    for filename in tqdm(annotation_files):
        filepath = os.path.join(annotation_dir, filename)
        try:
            # Load annotation image
            ann = np.array(Image.open(filepath))
            
            # Update label histogram
            unique_labels, label_counts = np.unique(ann, return_counts=True)
            for label, count in zip(unique_labels, label_counts):
                stats['label_histogram'][label] += count
            
            # Skip 255 (ignore) labels
            valid_mask = ann != 255
            ann_valid = ann[valid_mask]
            
            # Check for invalid labels
            unique_labels = np.unique(ann_valid)
            if len(unique_labels) > 0:
                max_label = unique_labels.max()
                invalid_labels = unique_labels[unique_labels >= num_classes]
                
                if len(invalid_labels) > 0:
                    stats['invalid_labels'] += len(invalid_labels)
                    problem_info = {
                        'file': filename,
                        'invalid_labels': invalid_labels.tolist(),
                        'invalid_counts': [np.sum(ann == label) for label in invalid_labels],
                        'total_pixels': ann.size,
                        'valid_labels': unique_labels[unique_labels < num_classes].tolist()
                    }
                    stats['problematic_files'].append(problem_info)
                    
                    # Print immediate warning for significant issues
                    if any(count > 1000 for count in problem_info['invalid_counts']):
                        print(f"\nWARNING - Major label issues in {filename}:")
                        print(f"Invalid labels (label: count): ", end="")
                        for label, count in zip(problem_info['invalid_labels'], 
                                             problem_info['invalid_counts']):
                            print(f"{label}: {count}", end=", ")
                        print(f"\nTotal pixels: {problem_info['total_pixels']}")
                
                stats['max_label_found'] = max(stats['max_label_found'], max_label)
                
                # Count pixels per class for valid labels
                for class_id in range(num_classes):
                    class_pixels[class_id] += np.sum(ann_valid == class_id)
                
        except Exception as e:
            print(f"\nError processing file {filename}: {str(e)}")
            continue
    
    # Print comprehensive analysis
    print("\n=== Label Analysis Summary ===")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Invalid label occurrences: {stats['invalid_labels']}")
    print(f"Maximum label value found: {stats['max_label_found']}")
    
    # Print label histogram
    print("\nLabel Distribution:")
    for label, count in enumerate(stats['label_histogram']):
        if count > 0:
            if label >= num_classes and label != 255:
                print(f"WARNING - Invalid label {label}: {count} pixels")
            elif label == 255:
                print(f"Ignore label (255): {count} pixels")
            else:
                print(f"Label {label}: {count} pixels")
    
    print("\nProblematic Files Summary:")
    if stats['problematic_files']:
        for i, file_info in enumerate(stats['problematic_files'][:5], 1):
            print(f"\n{i}. File: {file_info['file']}")
            print(f"   Invalid labels: {file_info['invalid_labels']}")
            print(f"   Invalid pixel counts: {file_info['invalid_counts']}")
            print(f"   Percentage invalid: {sum(file_info['invalid_counts'])/file_info['total_pixels']*100:.2f}%")
        
        if len(stats['problematic_files']) > 5:
            print(f"\n... and {len(stats['problematic_files']) - 5} more files with issues")
    else:
        print("No files with invalid labels found")
    
    # Check if we have any valid data
    if class_pixels.sum() == 0:
        raise ValueError("No valid pixel counts found in dataset")
    
    # Convert to frequencies
    total_pixels = class_pixels.sum()
    class_frequencies = class_pixels / total_pixels
    
    return class_frequencies, stats

def compute_class_weights(frequencies, method='median', beta=0.9999):
    """
    Compute class weights using different balancing methods with validation.
    
    Args:
        frequencies (np.ndarray): Class frequencies
        method (str): Weighting method ('median' or 'inverse')
        beta (float): Trimming parameter for inverse frequency
    
    Returns:
        np.ndarray: Class weights
    """
    # Validate input
    if np.any(frequencies < 0):
        raise ValueError("Negative frequencies detected")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Non-finite frequencies detected")
    
    if method == 'median':
        # Median frequency balancing
        median_freq = np.median(frequencies[frequencies > 0])  # Consider only non-zero frequencies
        weights = np.where(frequencies > 0, median_freq / frequencies, 1.0)
        
    elif method == 'inverse':
        # Inverse frequency with beta-trimming
        weights = np.where(frequencies > 0,
                          (1 - beta) / (1 - beta ** frequencies),
                          1.0)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Clip extreme values
    weights = np.clip(weights, 0.1, 10.0)
    
    # Normalize weights to have mean 1
    weights = weights / weights.mean()
    
    # Set weight for ignore label (class 255) to 0
    ignore_weight = np.zeros(1, dtype=weights.dtype)
    weights = np.concatenate([weights, ignore_weight])
    
    return weights

def generate_class_weights(data_root, save_path=None, method='median', beta=0.9999):
    """
    Generate and optionally save class weights for ADE20K dataset with validation.
    
    Args:
        data_root (str): Path to ADE20K dataset root
        save_path (str, optional): Path to save weights
        method (str): Weighting method ('median' or 'inverse')
        beta (float): Trimming parameter for inverse frequency method
    
    Returns:
        np.ndarray: Computed class weights
    """
    # Calculate class frequencies with validation
    frequencies, stats = calculate_class_frequencies(data_root)
    
    # Compute weights
    weights = compute_class_weights(frequencies, method=method, beta=beta)
    
    # Additional validation
    if np.any(~np.isfinite(weights)):
        raise ValueError("Generated weights contain non-finite values")
    
    # Save weights if path is provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, weights)
            print(f"Successfully saved class weights to {save_path}")
            
            # Print weight statistics
            print("\nWeight Statistics:")
            print(f"Shape: {weights.shape}")
            print(f"Min weight: {weights[:-1].min():.4f}")  # Exclude ignore label
            print(f"Max weight: {weights[:-1].max():.4f}")  # Exclude ignore label
            print(f"Mean weight: {weights[:-1].mean():.4f}")  # Exclude ignore label
            print(f"Ignore label weight: {weights[-1]:.4f}")
            
        except Exception as e:
            logging.error(f"Error saving weights to {save_path}: {str(e)}")
    
    return weights, stats

def load_class_weights(weights_path):
    """
    Load pre-computed class weights with validation.
    
    Args:
        weights_path (str): Path to the .npy file containing class weights
    
    Returns:
        np.ndarray: Class weights
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Class weights file not found: {weights_path}")
    
    try:
        weights = np.load(weights_path)
        
        # Validate loaded weights
        if not isinstance(weights, np.ndarray):
            raise ValueError("Loaded weights must be a numpy array")
        if np.any(~np.isfinite(weights)):
            raise ValueError("Loaded weights contain non-finite values")
        if np.any(weights[:-1] < 0):  # Exclude ignore label weight
            raise ValueError("Loaded weights contain negative values")
            
        return weights
        
    except Exception as e:
        raise ValueError(f"Error loading weights from {weights_path}: {str(e)}")

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate class weights for ADE20K dataset')
    parser.add_argument('--data-root', required=True, help='Path to ADE20K dataset root')
    parser.add_argument('--save-path', required=True, help='Path to save the weights')
    parser.add_argument('--method', default='median', choices=['median', 'inverse'],
                      help='Weighting method (default: median)')
    parser.add_argument('--beta', type=float, default=0.9999,
                      help='Beta parameter for inverse frequency method (default: 0.9999)')
    
    args = parser.parse_args()
    
    try:
        weights, stats = generate_class_weights(
            data_root=args.data_root,
            save_path=args.save_path,
            method=args.method,
            beta=args.beta
        )
        
        print("\nDataset Statistics:")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Invalid labels found: {stats['invalid_labels']}")
        print(f"Maximum label value found: {stats['max_label_found']}")
        print("\nClass Weights Shape:", weights.shape)
        print(f"Class Weight Stats (excluding ignore label):")
        print(f"- Min weight: {weights[:-1].min():.4f}")
        print(f"- Max weight: {weights[:-1].max():.4f}")
        print(f"- Mean weight: {weights[:-1].mean():.4f}")
        print(f"- Median weight: {np.median(weights[:-1]):.4f}")
        print(f"- Ignore label weight: {weights[-1]:.4f}")
        
    except Exception as e:
        logging.error(f"Error generating weights: {str(e)}")
        raise