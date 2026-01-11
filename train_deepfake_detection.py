"""
Deepfake Detection using EfficientNet-B4

Dataset: JamieWithofs/Deepfake-and-real-images from HuggingFace (140K images)
Model: EfficientNet-B4 (from scratch implementation)
Task: Binary classification (Fake vs Real)
  - Label 0: Fake (딥페이크 이미지)
  - Label 1: Real (진짜 이미지)

Training Pipeline:
1. Load dataset from HuggingFace
2. Data augmentation and preprocessing
3. Train EfficientNet-B4
4. Evaluation and visualization
5. Save best model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
import json

# Import our EfficientNet implementation
from efficientnet_from_scratch import efficientnet_b4, count_parameters


# ============================================================================
# 1. Dataset Preparation
# ============================================================================

class DeepfakeDataset(Dataset):
    """
    Custom Dataset for Deepfake Detection
    
    Wraps HuggingFace dataset with PyTorch Dataset interface
    """
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: HuggingFace dataset split
            transform: torchvision transforms
        """
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image and label
        image = item['image']
        label = item['label']  # 0: Fake, 1: Real
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=380, is_training=True):
    """
    Get data augmentation transforms
    
    Args:
        image_size: target image size (EfficientNet-B4 uses 380x380)
        is_training: whether for training or validation
    
    Returns:
        torchvision transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def balance_dataset(dataset, label_column='label'):
    """
    Balance dataset to have equal number of samples per class
    
    Args:
        dataset: HuggingFace dataset
        label_column: name of the label column
    
    Returns:
        balanced dataset
    """
    # Get all labels
    labels = dataset[label_column]
    
    # Count samples per class
    # label=0: Fake, label=1: Real
    fake_indices = [i for i, label in enumerate(labels) if label == 0]
    real_indices = [i for i, label in enumerate(labels) if label == 1]
    
    num_fake = len(fake_indices)
    num_real = len(real_indices)
    
    print(f"\n📊 Original distribution:")
    print(f"  Fake (label=0): {num_fake} ({num_fake/(num_fake+num_real)*100:.1f}%)")
    print(f"  Real (label=1): {num_real} ({num_real/(num_fake+num_real)*100:.1f}%)")
    
    # Balance to 1:1 ratio
    min_samples = min(num_fake, num_real)
    
    # Randomly sample equal number from each class
    import random
    random.seed(42)
    
    balanced_fake_indices = random.sample(fake_indices, min_samples)
    balanced_real_indices = random.sample(real_indices, min_samples)
    
    # Combine and shuffle
    balanced_indices = balanced_fake_indices + balanced_real_indices
    random.shuffle(balanced_indices)
    
    # Select balanced subset
    balanced_dataset = dataset.select(balanced_indices)
    
    print(f"\n✓ Balanced distribution:")
    print(f"  Fake (label=0): {min_samples} (50.0%)")
    print(f"  Real (label=1): {min_samples} (50.0%)")
    print(f"  Total: {len(balanced_dataset)}")
    
    return balanced_dataset


def load_deepfake_dataset(batch_size=16, num_workers=4, image_size=380):
    """
    Load Deepfake dataset from HuggingFace with balanced class distribution
    
    Dataset: JamieWithofs/Deepfake-and-real-images (140K images)
    - train: ~140K images
    - validation: validation split
    - test: test split
    
    Args:
        batch_size: batch size for DataLoader
        num_workers: number of worker processes
        image_size: target image size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 80)
    print("📦 Loading Deepfake Dataset from HuggingFace")
    print("=" * 80)
    
    # Load dataset from HuggingFace
    print("\n🔄 Downloading dataset (140K images, may take a few minutes)...")
    dataset = load_dataset("JamieWithofs/Deepfake-and-real-images")
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    # Balance the training set
    print("\n" + "=" * 80)
    print("⚖️  Balancing Training Dataset")
    print("=" * 80)
    train_dataset = balance_dataset(dataset['train'])
    
    # Balance validation set
    print("\n" + "=" * 80)
    print("⚖️  Balancing Validation Dataset")
    print("=" * 80)
    val_dataset = balance_dataset(dataset['validation'])
    
    # Balance test set
    print("\n" + "=" * 80)
    print("⚖️  Balancing Test Dataset")
    print("=" * 80)
    test_dataset = balance_dataset(dataset['test'])
    
    # Verify distributions
    print("\n" + "=" * 80)
    print("📊 Final Dataset Splits")
    print("=" * 80)
    
    def print_distribution(dataset, name):
        labels = dataset['label']
        num_fake = sum(1 for l in labels if l == 0)
        num_real = sum(1 for l in labels if l == 1)
        total = len(labels)
        print(f"\n{name}:")
        print(f"  Total: {total}")
        print(f"  Fake (label=0): {num_fake} ({num_fake/total*100:.1f}%)")
        print(f"  Real (label=1): {num_real} ({num_real/total*100:.1f}%)")
    
    print_distribution(train_dataset, "Train")
    print_distribution(val_dataset, "Validation")
    print_distribution(test_dataset, "Test")
    
    # Create PyTorch datasets
    print("\n" + "=" * 80)
    print("🔧 Creating DataLoaders")
    print("=" * 80)
    
    train_transforms = get_transforms(image_size, is_training=True)
    val_transforms = get_transforms(image_size, is_training=False)
    
    train_pytorch_dataset = DeepfakeDataset(train_dataset, train_transforms)
    val_pytorch_dataset = DeepfakeDataset(val_dataset, val_transforms)
    test_pytorch_dataset = DeepfakeDataset(test_dataset, val_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_pytorch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_pytorch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders created!")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("=" * 80)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 2. Training Functions
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: EfficientNet model
        train_loader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: cuda or cpu
        epoch: current epoch number
    
    Returns:
        avg_loss, avg_acc
    """
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / batch_size
        
        # Update metrics
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, epoch=None):
    """
    Validate the model
    
    Args:
        model: EfficientNet model
        val_loader: validation data loader
        criterion: loss function
        device: cuda or cpu
        epoch: current epoch number (optional)
    
    Returns:
        avg_loss, avg_acc, all_preds, all_labels
    """
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / batch_size
            
            # Update metrics
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy, batch_size)
            
            # Store predictions
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
    
    return losses.avg, accuracies.avg, all_preds, all_labels


# ============================================================================
# 3. Training Pipeline
# ============================================================================

def train_deepfake_detector(
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    image_size=380,
    save_dir='checkpoints',
    device=None
):
    """
    Complete training pipeline for Deepfake detection
    
    Args:
        num_epochs: number of training epochs
        batch_size: batch size
        learning_rate: initial learning rate
        weight_decay: L2 regularization
        image_size: input image size (380 for B4)
        save_dir: directory to save checkpoints
        device: cuda or cpu
    
    Returns:
        model, history
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("🚀 Starting Deepfake Detection Training")
    print("=" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"  Model: EfficientNet-B4")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Image size: {image_size}x{image_size}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"efficientnet_b4_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, test_loader = load_deepfake_dataset(
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size
    )
    
    # Create model
    print("\n🏗️  Building Model...")
    model = efficientnet_b4(num_classes=2)  # Binary classification
    model = model.to(device)
    
    num_params = count_parameters(model) / 1e6
    print(f"✓ Model created: EfficientNet-B4")
    print(f"✓ Parameters: {num_params:.2f}M")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print("\n" + "=" * 80)
    print("🔥 Training Started")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")
        print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            
            print(f"   ⭐ New best model saved! (Val Acc: {val_acc:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Training completed
    print("\n" + "=" * 80)
    print("✅ Training Completed!")
    print("=" * 80)
    print(f"\n🏆 Best Results:")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val Acc: {best_val_acc:.4f}")
    
    # Load best model for testing
    print("\n🧪 Testing on Test Set...")
    best_checkpoint = torch.load(os.path.join(experiment_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Acc:  {test_acc:.4f}")
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n📈 Classification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=['Fake', 'Real'],  # 0: Fake, 1: Real
        digits=4
    ))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"              Predicted")
    print(f"              Fake  Real")
    print(f"   Fake      {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"   Real      {cm[1][0]:5d} {cm[1][1]:5d}")
    
    # Save history
    history_path = os.path.join(experiment_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training curves
    plot_training_curves(history, experiment_dir)
    
    print(f"\n💾 All results saved to: {experiment_dir}")
    print("=" * 80)
    
    return model, history


# ============================================================================
# 4. Visualization
# ============================================================================

def plot_training_curves(history, save_dir):
    """
    Plot training curves
    
    Args:
        history: training history dictionary
        save_dir: directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate curve
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {plot_path}")
    plt.close()


# ============================================================================
# 5. Inference Function
# ============================================================================

def predict_image(model, image_path, device, image_size=380):
    """
    Predict if an image is real or fake
    
    Args:
        model: trained EfficientNet model
        image_path: path to image file
        device: cuda or cpu
        image_size: input image size
    
    Returns:
        prediction (0: Real, 1: Fake), confidence
    """
    model.eval()
    
    # Load and preprocess image
    transform = get_transforms(image_size, is_training=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return predicted.item(), confidence.item()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Training configuration
    config = {
        'num_epochs': 20,
        'batch_size': 16,  # Adjust based on GPU memory
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'image_size': 380,  # EfficientNet-B4 default
        'save_dir': 'checkpoints_deepfake',
    }
    
    print("\n" + "=" * 80)
    print("🎯 Deepfake Detection with EfficientNet-B4")
    print("=" * 80)
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Train model
    model, history = train_deepfake_detector(**config)
    
    print("\n" + "=" * 80)
    print("🎉 All Done!")
    print("=" * 80)
    print("\n💡 To use the trained model for inference:")
    print("   from train_deepfake_detection import predict_image")
    print("   prediction, confidence = predict_image(model, 'image.jpg', device)")
    print("   print(f'Prediction: {'Fake' if prediction == 1 else 'Real'}')")
    print("   print(f'Confidence: {confidence:.2%}')")
    print("=" * 80)

