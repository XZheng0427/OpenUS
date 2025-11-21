import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json


class FetalPlanesDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, target_transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Set the correct directory based on split
        if split == 'train':
            self.data_dir = self.data_root / 'training_set'
        elif split == 'test':
            self.data_dir = self.data_root / 'test_set'
        else:
            raise ValueError("split must be either 'train' or 'test'")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        # Get class names from directory structure
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.class_names)
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples")
        print(f"Classes ({self.num_classes}): {self.class_names}")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image paths and their corresponding labels."""
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    self.samples.append((str(img_path), class_idx))
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset."""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\nClass distribution for {self.split} set:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a default image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def get_class_weights(self):
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.samples)
        weights = []
        
        for i in range(self.num_classes):
            if i in class_counts:
                weight = total_samples / (self.num_classes * class_counts[i])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def save_class_mapping(self, save_path):
        mapping = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }
        
        with open(save_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Class mapping saved to {save_path}")


def get_transforms(split='train', image_size=224):
    if split == 'train':
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        # Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    return transform


def create_data_loaders(data_root, batch_size=32, num_workers=4, image_size=224, 
                       pin_memory=True, shuffle_train=True):
    
    train_transform = get_transforms('train', image_size)
    test_transform = get_transforms('test', image_size)
    
    # Create datasets
    train_dataset = FetalPlanesDataset(
        data_root=data_root,
        split='train',
        transform=train_transform
    )
    
    test_dataset = FetalPlanesDataset(
        data_root=data_root,
        split='test',
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def get_sample_weights(dataset):
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    sample_weights = []
    for _, label in dataset.samples:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)
    
    return torch.FloatTensor(sample_weights)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Fetal Planes Dataset')
    parser.add_argument('--data_root', 
                       default='./FETAL_PLANES_ZENODO',
                       help='Root directory containing training_set and test_set')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    print("Creating data loaders...")
    try:
        train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=2  # Reduce for testing
        )
        
        print(f"\nDataset created successfully!")
        print(f"Training batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Save class mapping
        mapping_path = os.path.join(args.data_root, 'class_mapping.json')
        train_dataset.save_class_mapping(mapping_path)
        
        # Test loading a batch
        print("\nTesting data loading...")
        train_iter = iter(train_loader)
        images, labels = next(train_iter)
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample labels: {labels[:5]}")
        
        # Print class weights
        class_weights = train_dataset.get_class_weights()
        print(f"\nClass weights: {class_weights}")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Make sure you have run the dataset creation script first!") 