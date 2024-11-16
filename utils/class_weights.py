import numpy as np
import os

def setup_weights(cfg):
    """Setup class weights for training."""
    try:
        weights = None
        ratio_file = 'ADEChallengeData2016/objectInfo150.txt'
        
        if os.path.exists(ratio_file):
            ratios, class_samples = load_class_ratios_from_file(ratio_file)
            weights = compute_class_weights(ratios, class_samples)
            
            # Update config - handle list of losses
            if weights is not None and len(weights) == 151:
                if isinstance(cfg.model.decode_head.loss_decode, list):
                    for loss in cfg.model.decode_head.loss_decode:
                        if loss['type'] == 'CrossEntropyLoss':
                            loss['class_weight'] = weights.tolist()
                            break
                else:
                    cfg.model.decode_head.loss_decode.class_weight = weights.tolist()
            else:
                print("Invalid weights shape, expected 151 classes")
                
        else:
            print(f"objectInfo150.txt not found at {ratio_file}")
            return cfg
            
    except Exception as e:
        print(f"Error setting up weights: {str(e)}")
        return cfg
    
    return cfg

def load_class_ratios_from_file(ratio_file_path):
    """Load class ratios from the ADE20K objectInfo150.txt file."""
    ratios = np.zeros(151, dtype=np.float32)
    class_samples = np.zeros(151, dtype=np.int32)
    
    with open(ratio_file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                idx = int(parts[0]) - 1
                ratio = float(parts[1])
                train_samples = int(parts[2])
                ratios[idx] = ratio
                class_samples[idx] = train_samples
    
    return ratios, class_samples

def compute_class_weights(ratios, class_samples):
    """Compute balanced class weights with improved scaling."""
    # Use log scaling for frequency weights to compress the range
    freq_weights = np.where(ratios > 0,
                          1 / np.log1p(ratios * 100),  # Multiply by 100 to handle small ratios better
                          1.0)
    
    # Log scaling for sample counts with better normalization
    max_samples = np.max(class_samples)
    sample_weights = np.where(class_samples > 0,
                            1 / np.log1p(class_samples / max_samples * 1000),
                            1.0)
    
    # Combine weights with balanced contribution
    weights = freq_weights * np.sqrt(sample_weights)  # Reduce sample count influence
    
    # Normalize more gently
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-5)
    weights = weights * 1.5 + 0.1  # Adjust range to [0.1, 1.6]
    
    # Normalize to mean 1
    weights = weights / weights[weights > 0].mean()
    
    # Clip with tighter bounds
    weights = np.clip(weights, 0.1, 5.0)
    
    # Final normalization
    weights = weights / weights[weights > 0].mean()
    
    # Set ignore label weight to 0
    weights[-1] = 0
    
    return weights

if __name__ == '__main__':
    data_file = 'ADEChallengeData2016/objectInfo150.txt'
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        exit(1)
        
    ratios, class_samples = load_class_ratios_from_file(data_file)
    weights = compute_class_weights(ratios, class_samples)
    
    print(f"Computed weights for {len(weights)-1} classes")
    print(f"Min weight: {weights[:-1].min():.4f}")
    print(f"Max weight: {weights[:-1].max():.4f}")
    print(f"Median weight: {np.median(weights[:-1]):.4f}")
    
    print("\nDetailed weights:")
    for i, weight in enumerate(weights[:-1]):
        print(f"Class {i+1:3d}: {weight:.4f}")