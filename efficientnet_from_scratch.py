"""
EfficientNet Implementation from Scratch in PyTorch

Reference: 
- Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- https://arxiv.org/abs/1905.11946

Key Components:
1. MBConv (Mobile Inverted Bottleneck Convolution)
2. Squeeze-and-Excitation (SE) blocks
3. Swish activation function
4. Compound Scaling (depth, width, resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


# ============================================================================
# 1. Activation Functions
# ============================================================================

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Also known as SiLU (Sigmoid Linear Unit)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


# ============================================================================
# 2. Squeeze-and-Excitation Block
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Architecture:
    Input → Global Average Pool → FC → ReLU → FC → Sigmoid → Scale Input
    
    Purpose:
    - Channel-wise attention mechanism
    - Recalibrates channel-wise feature responses
    """
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(in_channels, squeezed_channels, 1),  # Squeeze
            Swish(),
            nn.Conv2d(squeezed_channels, in_channels, 1),  # Excitation
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) - recalibrated features
        """
        scale = self.se(x)  # (B, C, 1, 1)
        return x * scale    # Channel-wise scaling


# ============================================================================
# 3. MBConv Block (Mobile Inverted Bottleneck Convolution)
# ============================================================================

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block
    
    Architecture:
    Input → [Expansion (1x1 Conv)] → Depthwise Conv → SE Block → 
    Projection (1x1 Conv) → [Stochastic Depth] → Output
    
    Key Features:
    - Inverted residual structure (expand → depthwise → project)
    - Squeeze-and-Excitation for channel attention
    - Skip connection when stride=1 and in_channels == out_channels
    - Stochastic depth for regularization
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        
        # Skip connection only when stride=1 and same dimensions
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Hidden dimension (expansion)
        hidden_dim = in_channels * expand_ratio
        
        # Build layers
        layers = []
        
        # 1. Expansion phase (only if expand_ratio != 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])
        
        # 2. Depthwise convolution phase
        layers.extend([
            nn.Conv2d(
                hidden_dim, 
                hidden_dim, 
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,  # Depthwise: each channel convolved separately
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # 3. Squeeze-and-Excitation
        self.se = SEBlock(hidden_dim, se_ratio)
        
        # 4. Projection phase (pointwise convolution)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            out: (B, out_channels, H', W')
        """
        identity = x
        
        # Main path
        x = self.conv(x)
        x = self.se(x)
        x = self.project(x)
        
        # Skip connection with stochastic depth
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                x = self.drop_connect(x)
            x = x + identity
        
        return x
    
    def drop_connect(self, x):
        """
        Stochastic Depth (Drop Connect)
        Randomly drop the entire residual branch
        """
        keep_prob = 1 - self.drop_connect_rate
        
        # Random tensor with same shape as batch
        random_tensor = keep_prob + torch.rand(
            (x.shape[0], 1, 1, 1),
            dtype=x.dtype,
            device=x.device
        )
        binary_tensor = torch.floor(random_tensor)
        
        # Scale output to maintain expected value
        output = x / keep_prob * binary_tensor
        
        return output


# ============================================================================
# 4. EfficientNet Architecture
# ============================================================================

class EfficientNet(nn.Module):
    """
    EfficientNet: Scalable Convolutional Neural Network
    
    Architecture:
    Stem → MBConv Blocks (7 stages) → Head → FC
    
    Compound Scaling:
    - depth: number of layers (repeats)
    - width: number of channels
    - resolution: input image size
    
    Formula:
    depth = alpha ^ phi
    width = beta ^ phi
    resolution = gamma ^ phi
    where alpha * beta^2 * gamma^2 ≈ 2
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2
    ):
        """
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier (beta)
            depth_mult: Depth multiplier (alpha)
            dropout_rate: Dropout rate before final FC
            drop_connect_rate: Drop connect rate for MBConv blocks
        """
        super().__init__()
        
        # Building blocks configuration
        # [expand_ratio, channels, repeats, stride, kernel_size]
        self.block_configs = [
            # Stage 1
            [1, 16, 1, 1, 3],
            # Stage 2
            [6, 24, 2, 2, 3],
            # Stage 3
            [6, 40, 2, 2, 5],
            # Stage 4
            [6, 80, 3, 2, 3],
            # Stage 5
            [6, 112, 3, 1, 5],
            # Stage 6
            [6, 192, 4, 2, 5],
            # Stage 7
            [6, 320, 1, 1, 3],
        ]
        
        # Stem
        out_channels = self._round_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        
        total_blocks = sum([config[2] for config in self.block_configs])
        block_idx = 0
        
        for expand_ratio, channels, repeats, stride, kernel_size in self.block_configs:
            out_channels = self._round_channels(channels, width_mult)
            num_repeats = self._round_repeats(repeats, depth_mult)
            
            for i in range(num_repeats):
                # Calculate drop connect rate (linearly increase)
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                # First block of each stage may have stride > 1
                block_stride = stride if i == 0 else 1
                
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_connect_rate=drop_rate
                    )
                )
                
                in_channels = out_channels
                block_idx += 1
        
        # Head
        final_channels = self._round_channels(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish()
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _round_channels(self, channels: int, width_mult: float, divisor: int = 8) -> int:
        """
        Round number of channels to nearest divisor (for hardware efficiency)
        """
        channels *= width_mult
        new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
        
        # Make sure rounding doesn't decrease by more than 10%
        if new_channels < 0.9 * channels:
            new_channels += divisor
        
        return int(new_channels)
    
    def _round_repeats(self, repeats: int, depth_mult: float) -> int:
        """
        Round number of block repeats based on depth multiplier
        """
        return int(math.ceil(depth_mult * repeats))
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 3, H, W) - input images
            return_features: if True, return intermediate features
        
        Returns:
            out: (B, num_classes) - class logits
            features: (optional) list of intermediate features
        """
        features = []
        
        # Stem
        x = self.stem(x)
        if return_features:
            features.append(x)
        
        # MBConv blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if return_features and idx in [1, 3, 5, 11, 15]:  # Key stages
                features.append(x)
        
        # Head
        x = self.head(x)
        if return_features:
            features.append(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if return_features:
            return x, features
        return x
    
    def extract_features(self, x):
        """
        Extract features without classification head
        Useful for transfer learning
        """
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============================================================================
# 5. EfficientNet Variants (B0 ~ B7)
# ============================================================================

def efficientnet_b0(num_classes: int = 1000, pretrained: bool = False):
    """
    EfficientNet-B0
    - Resolution: 224x224
    - Parameters: 5.3M
    - Top-1 Accuracy: 77.1%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2
    )
    return model


def efficientnet_b1(num_classes: int = 1000):
    """
    EfficientNet-B1
    - Resolution: 240x240
    - Parameters: 7.8M
    - Top-1 Accuracy: 79.1%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.1,
        dropout_rate=0.2
    )
    return model


def efficientnet_b2(num_classes: int = 1000):
    """
    EfficientNet-B2
    - Resolution: 260x260
    - Parameters: 9.2M
    - Top-1 Accuracy: 80.1%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.1,
        depth_mult=1.2,
        dropout_rate=0.3
    )
    return model


def efficientnet_b3(num_classes: int = 1000):
    """
    EfficientNet-B3
    - Resolution: 300x300
    - Parameters: 12M
    - Top-1 Accuracy: 81.6%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.2,
        depth_mult=1.4,
        dropout_rate=0.3
    )
    return model


def efficientnet_b4(num_classes: int = 1000):
    """
    EfficientNet-B4
    - Resolution: 380x380
    - Parameters: 19M
    - Top-1 Accuracy: 82.9%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.4,
        depth_mult=1.8,
        dropout_rate=0.4
    )
    return model


def efficientnet_b5(num_classes: int = 1000):
    """
    EfficientNet-B5
    - Resolution: 456x456
    - Parameters: 30M
    - Top-1 Accuracy: 83.6%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.6,
        depth_mult=2.2,
        dropout_rate=0.4
    )
    return model


def efficientnet_b6(num_classes: int = 1000):
    """
    EfficientNet-B6
    - Resolution: 528x528
    - Parameters: 43M
    - Top-1 Accuracy: 84.0%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=1.8,
        depth_mult=2.6,
        dropout_rate=0.5
    )
    return model


def efficientnet_b7(num_classes: int = 1000):
    """
    EfficientNet-B7
    - Resolution: 600x600
    - Parameters: 66M
    - Top-1 Accuracy: 84.3%
    """
    model = EfficientNet(
        num_classes=num_classes,
        width_mult=2.0,
        depth_mult=3.1,
        dropout_rate=0.5
    )
    return model


# ============================================================================
# 6. Model Summary and Testing
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test EfficientNet implementation"""
    print("=" * 80)
    print("🧪 Testing EfficientNet Implementation")
    print("=" * 80)
    
    # Test different variants
    variants = [
        ('B0', efficientnet_b0, 224),
        ('B1', efficientnet_b1, 240),
        ('B2', efficientnet_b2, 260),
        ('B3', efficientnet_b3, 300),
        ('B4', efficientnet_b4, 380),
    ]
    
    for name, model_fn, resolution in variants:
        print(f"\n{'='*60}")
        print(f"Testing EfficientNet-{name}")
        print(f"{'='*60}")
        
        model = model_fn(num_classes=1000)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, resolution, resolution)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Print stats
        num_params = count_parameters(model) / 1e6
        print(f"✓ Input shape:  {tuple(x.shape)}")
        print(f"✓ Output shape: {tuple(output.shape)}")
        print(f"✓ Parameters:   {num_params:.2f}M")
        print(f"✓ Expected:     (2, 1000)")
        
        assert output.shape == (batch_size, 1000), "Output shape mismatch!"
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


def visualize_architecture():
    """Visualize model architecture"""
    print("\n" + "=" * 80)
    print("📊 EfficientNet-B0 Architecture")
    print("=" * 80)
    
    model = efficientnet_b0(num_classes=1000)
    
    print("\n[Stem]")
    print(f"  Conv2d(3 → 32, k=3, s=2) + BN + Swish")
    print(f"  Output: (B, 32, 112, 112)")
    
    print("\n[MBConv Blocks]")
    stage_names = [
        "Stage 1: MBConv1 (k=3, 16 channels,  1 repeat, s=1)",
        "Stage 2: MBConv6 (k=3, 24 channels,  2 repeats, s=2)",
        "Stage 3: MBConv6 (k=5, 40 channels,  2 repeats, s=2)",
        "Stage 4: MBConv6 (k=3, 80 channels,  3 repeats, s=2)",
        "Stage 5: MBConv6 (k=5, 112 channels, 3 repeats, s=1)",
        "Stage 6: MBConv6 (k=5, 192 channels, 4 repeats, s=2)",
        "Stage 7: MBConv6 (k=3, 320 channels, 1 repeat,  s=1)",
    ]
    
    for stage in stage_names:
        print(f"  {stage}")
    
    print("\n[Head]")
    print(f"  Conv2d(320 → 1280, k=1) + BN + Swish")
    print(f"  AdaptiveAvgPool2d(1)")
    print(f"  Dropout(0.2)")
    print(f"  Linear(1280 → 1000)")
    
    print("\n[Total]")
    print(f"  Parameters: {count_parameters(model)/1e6:.2f}M")
    print("=" * 80)


# ============================================================================
# 7. Training Example
# ============================================================================

def train_example():
    """
    Example training loop for EfficientNet
    """
    print("\n" + "=" * 80)
    print("🔥 Training Example")
    print("=" * 80)
    
    # Hyperparameters
    num_classes = 10  # e.g., CIFAR-10
    batch_size = 8
    num_epochs = 5
    learning_rate = 0.001
    
    # Model
    model = efficientnet_b0(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dummy data (replace with real DataLoader)
    print(f"\n📊 Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Num classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    
    print("\n🔄 Training loop:")
    for epoch in range(num_epochs):
        # Dummy batch
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print
        acc = (outputs.argmax(1) == labels).float().mean()
        print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
    
    print("\n✅ Training completed!")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Test implementation
    test_model()
    
    # Visualize architecture
    visualize_architecture()
    
    # Training example
    train_example()
    
    print("\n" + "=" * 80)
    print("🎉 EfficientNet Implementation Complete!")
    print("=" * 80)
    print("\n💡 Usage:")
    print("  from efficientnet_from_scratch import efficientnet_b0")
    print("  model = efficientnet_b0(num_classes=10)")
    print("  output = model(torch.randn(1, 3, 224, 224))")
    print("=" * 80)

