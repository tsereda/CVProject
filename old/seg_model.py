# seg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import timm
import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):
    """Segmentation model for comparing modern backbones and decoder architectures.
    
    Key Features:
    - Backbone options: ConvNeXt, PVT, Swin (via timm)
    - Decoder options: SMP implementations or custom
    - Feature extraction for analysis
    
    Example:
        model = SegmentationModel(
            num_classes=150,
            backbone='convnext_tiny',      # or 'pvt_v2_b0', 'swin_tiny_patch4_window7_224'
            decoder='smp_fpn',             # or 'smp_unet', 'custom_fpn', etc.
        )
    """
    VALID_BACKBONES = {
        'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base'],
        'pvt': ['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2'],
        'swin': ['swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224']
    }
    
    VALID_DECODERS = {
        'smp': ['smp_fpn', 'smp_unet', 'smp_deeplabv3plus'],
        'custom': ['custom_fpn']  # Expand as we implement more
    }

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'convnext_tiny',
        decoder: str = 'smp_fpn',
        pretrained: bool = True,
        output_hidden_features: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.decoder_name = decoder
        self.output_hidden_features = output_hidden_features

        # Validate configurations
        self._validate_backbone(backbone)
        self._validate_decoder(decoder)

        # Initialize model based on decoder type
        if decoder.startswith('smp_'):
            self.model = self._create_smp_model(pretrained)
            self.using_smp = True
        else:
            self.encoder = self._create_encoder(pretrained)
            self.decoder = self._create_custom_decoder()
            self.head = nn.Conv2d(self.decoder.out_channels, num_classes, 1)
            self.using_smp = False

        # Feature storage for analysis
        self.features = {}
        if output_hidden_features:
            self._register_hooks()

    def _validate_backbone(self, backbone: str):
        """Ensure backbone is supported."""
        valid = any(
            backbone in options 
            for options in self.VALID_BACKBONES.values()
        )
        if not valid:
            raise ValueError(
                f"Backbone {backbone} not supported. "
                f"Valid options: {self.VALID_BACKBONES}"
            )

    def _validate_decoder(self, decoder: str):
        """Ensure decoder is supported."""
        valid = any(
            decoder in options 
            for options in self.VALID_DECODERS.values()
        )
        if not valid:
            raise ValueError(
                f"Decoder {decoder} not supported. "
                f"Valid options: {self.VALID_DECODERS}"
            )

    def _create_smp_model(self, pretrained: bool) -> nn.Module:
        """Create model using SMP."""
        decoder_name = self.decoder_name.replace('smp_', '')
        return smp.create_model(
            arch=decoder_name,
            encoder_name=self.backbone_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=self.num_classes
        )

    def _create_encoder(self, pretrained: bool) -> nn.Module:
        """Create backbone from timm."""
        return timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )

    def _create_custom_decoder(self) -> nn.Module:
        """Create custom decoder implementation."""
        if self.decoder_name == 'custom_fpn':
            raise NotImplementedError("Custom FPN to be implemented")
        raise ValueError(f"Unknown custom decoder: {self.decoder_name}")

    def _register_hooks(self):
        """Register hooks for feature extraction."""
        def hook(name):
            def _hook(module, input, output):
                self.features[name] = output
            return _hook

        if self.using_smp:
            # Register on SMP encoder stages
            for name, module in self.model.encoder.named_children():
                if isinstance(module, nn.Sequential):
                    module.register_forward_hook(hook(f'encoder_{name}'))
        else:
            # Register on custom implementation
            for name, module in self.encoder.named_children():
                if 'layer' in name or 'stage' in name:
                    module.register_forward_hook(hook(f'encoder_{name}'))

    def get_backbone_info(self) -> Dict:
        """Get information about backbone architecture."""
        if self.using_smp:
            return {
                'name': self.backbone_name,
                'encoder_channels': self.model.encoder.out_channels,
                'decoder_channels': getattr(self.model.decoder, 'out_channels', None)
            }
        else:
            # Get info from custom implementation
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = self.encoder(dummy)
            return {
                'name': self.backbone_name,
                'encoder_channels': [f.shape[1] for f in features]
            }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with optional feature extraction."""
        # Clear previous features
        if self.output_hidden_features:
            self.features.clear()

        # Forward pass
        if self.using_smp:
            out = self.model(x)
        else:
            features = self.encoder(x)
            decoded = self.decoder(features)
            out = self.head(decoded)

        # Prepare output
        output = {'out': out}
        if self.output_hidden_features:
            output['features'] = self.features
        
        return output

def create_model(num_classes: int, **kwargs) -> SegmentationModel:
    """Factory function for creating segmentation models."""
    return SegmentationModel(num_classes=num_classes, **kwargs)

# Example Usage
if __name__ == '__main__':
    # Compare different configurations
    model_configs = [
        {'backbone': 'convnext_tiny', 'decoder': 'smp_fpn'},
        {'backbone': 'pvt_v2_b0', 'decoder': 'smp_fpn'},
        {'backbone': 'swin_tiny_patch4_window7_224', 'decoder': 'smp_fpn'}
    ]
    
    # Test each configuration
    for config in model_configs:
        model = create_model(num_classes=150, **config)
        print(f"\nTesting {config}")
        
        # Get model info
        info = model.get_backbone_info()
        print(f"Backbone channels: {info['encoder_channels']}")
        
        # Test forward pass
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        print(f"Output shape: {out['out'].shape}")