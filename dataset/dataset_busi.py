import os
import shutil
import random
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def create_directory_structure(base_path):
    splits = ['training', 'validation', 'test']
    categories = ['benign', 'malignant', 'normal']
    
    for split in splits:
        for category in categories:
            dir_path = os.path.join(base_path, split, category)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def group_image_mask_pairs(all_files):
    pairs = []
    mask_files = set()
    
    # First, identify all mask files
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if '_mask.' in filename.lower():
            mask_files.add(file_path)
    
    # Then, find corresponding image files for each mask
    for file_path in all_files:
        if file_path in mask_files:
            continue  # Skip mask files in this loop
            
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        
        # Look for corresponding mask file
        possible_mask_names = [
            f"{name_without_ext}_mask{ext}",
            f"{name_without_ext}_mask.png",  # Common case where masks are always PNG
        ]
        
        mask_path = None
        for possible_mask in possible_mask_names:
            possible_mask_path = os.path.join(os.path.dirname(file_path), possible_mask)
            if possible_mask_path in mask_files:
                mask_path = possible_mask_path
                break
        
        if mask_path:
            pairs.append((file_path, mask_path))
        else:
            print(f"Warning: No mask found for image {filename}")
    
    return pairs

def split_dataset(source_path, destination_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    categories = ['benign', 'malignant', 'normal']
    
    # Create directory structure
    create_directory_structure(destination_path)
    
    for category in categories:
        category_path = os.path.join(source_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Category folder {category_path} does not exist. Skipping...")
            continue
        
        # Get all files in the category folder
        file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(category_path, pattern)))
        
        if not all_files:
            print(f"Warning: No image files found in {category_path}")
            continue
        
        # Group image-mask pairs
        image_mask_pairs = group_image_mask_pairs(all_files)
        
        if not image_mask_pairs:
            print(f"Warning: No valid image-mask pairs found in {category_path}")
            continue
        
        # Shuffle the pairs randomly
        random.shuffle(image_mask_pairs)
        
        # Calculate split indices
        total_pairs = len(image_mask_pairs)
        train_end = int(total_pairs * train_ratio)
        val_end = train_end + int(total_pairs * val_ratio)
        
        # Split pairs
        train_pairs = image_mask_pairs[:train_end]
        val_pairs = image_mask_pairs[train_end:val_end]
        test_pairs = image_mask_pairs[val_end:]
        
        print(f"\nCategory: {category}")
        print(f"Total image-mask pairs: {total_pairs}")
        print(f"Training: {len(train_pairs)} pairs ({len(train_pairs)*2} files)")
        print(f"Validation: {len(val_pairs)} pairs ({len(val_pairs)*2} files)")
        print(f"Test: {len(test_pairs)} pairs ({len(test_pairs)*2} files)")
        
        # Copy pairs to respective directories
        for pairs, split_name in [(train_pairs, 'training'), 
                                 (val_pairs, 'validation'), 
                                 (test_pairs, 'test')]:
            
            destination_dir = os.path.join(destination_path, split_name, category)
            
            for image_path, mask_path in pairs:
                # Copy image file
                image_filename = os.path.basename(image_path)
                destination_image = os.path.join(destination_dir, image_filename)
                shutil.copy2(image_path, destination_image)
                
                # Copy mask file
                mask_filename = os.path.basename(mask_path)
                destination_mask = os.path.join(destination_dir, mask_filename)
                shutil.copy2(mask_path, destination_mask)
            
            print(f"Copied {len(pairs)} pairs ({len(pairs)*2} files) to {destination_dir}")


class BUSIDataset(Dataset):
    def __init__(self, data_root, split='training', transform=None, mask_transform=None, 
                 categories=['benign', 'malignant', 'normal']):
        self.root_dir = data_root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.categories = categories
        
        # Create label mapping
        self.label_map = {category: idx for idx, category in enumerate(categories)}
        
        # Load all image-mask pairs
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        for category in categories:
            count = sum(1 for _, _, label in self.samples if label == self.label_map[category])
            print(f"  {category}: {count} samples")
    
    def _load_samples(self):
        """Load all image-mask pairs from the split directory."""
        samples = []
        
        for category in self.categories:
            category_dir = os.path.join(self.root_dir, self.split, category)
            
            if not os.path.exists(category_dir):
                print(f"Warning: Directory {category_dir} does not exist")
                continue
            
            # Get all files in the category directory
            file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            all_files = []
            for pattern in file_patterns:
                all_files.extend(glob.glob(os.path.join(category_dir, pattern)))
            
            # Group into image-mask pairs
            pairs = self._group_files_into_pairs(all_files)
            
            # Add to samples with label
            label = self.label_map[category]
            for image_path, mask_path in pairs:
                samples.append((image_path, mask_path, label))
        
        return samples
    
    def _group_files_into_pairs(self, all_files):
        """Group files into image-mask pairs."""
        pairs = []
        mask_files = set()
        
        # Identify mask files
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if '_mask.' in filename.lower():
                mask_files.add(file_path)
        
        # Find corresponding image files
        for file_path in all_files:
            if file_path in mask_files:
                continue
                
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            
            # Look for corresponding mask
            possible_mask_names = [
                f"{name_without_ext}_mask{ext}",
                f"{name_without_ext}_mask.png",
            ]
            
            mask_path = None
            for possible_mask in possible_mask_names:
                possible_mask_path = os.path.join(os.path.dirname(file_path), possible_mask)
                if possible_mask_path in mask_files:
                    mask_path = possible_mask_path
                    break
            
            if mask_path:
                pairs.append((file_path, mask_path))
        
        return pairs
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, mask_path, label = self.samples[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, label


def get_default_transforms(image_size=(224, 224), augment=False):
    if augment:
        # Training transforms with augmentation
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
    else:
        # Validation/Test transforms without augmentation
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    
    return image_transform, mask_transform

def create_data_loaders(dataset_root, batch_size=16, image_size=(256, 256), num_workers=4):
    # Get transforms
    train_img_transform, train_mask_transform = get_default_transforms(image_size, augment=True)
    val_img_transform, val_mask_transform = get_default_transforms(image_size, augment=False)
    
    # Create datasets
    train_dataset = BUSIDataset(
        root_dir=dataset_root,
        split='training',
        transform=train_img_transform,
        mask_transform=train_mask_transform
    )
    
    val_dataset = BUSIDataset(
        root_dir=dataset_root,
        split='validation',
        transform=val_img_transform,
        mask_transform=val_mask_transform
    )
    
    test_dataset = BUSIDataset(
        root_dir=dataset_root,
        split='test',
        transform=val_img_transform,
        mask_transform=val_mask_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return data_loaders

def demo_dataset_usage():
    dataset_root = "./BUSI/BUSI_split"
    
    print("Creating data loaders...")
    data_loaders = create_data_loaders(dataset_root, batch_size=4, image_size=(256, 256))
    
    print("\nDataset sizes:")
    for split, loader in data_loaders.items():
        print(f"{split}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    print("\nSample batch from training set:")
    train_loader = data_loaders['train']
    sample_batch = next(iter(train_loader))
    
    print(f"Image batch shape: {sample_batch['image'].shape}")
    print(f"Mask batch shape: {sample_batch['mask'].shape}")
    print(f"Labels: {sample_batch['label']}")
    print(f"Image paths: {sample_batch['image_path'][:2]}...")  # Show first 2 paths
    
    return data_loaders

def main():
    random.seed(42)
    
    # Define paths
    source_dataset_path = "./BUSI/Dataset_BUSI_with_GT"
    destination_dataset_path = "./BUSI/BUSI_split"
    
    # Check if source directory exists
    if not os.path.exists(source_dataset_path):
        print(f"Error: Source directory {source_dataset_path} does not exist!")
        return
    
    # Check if source has the required subdirectories
    required_categories = ['benign', 'malignant', 'normal']
    missing_categories = []
    
    for category in required_categories:
        category_path = os.path.join(source_dataset_path, category)
        if not os.path.exists(category_path):
            missing_categories.append(category)
    
    if missing_categories:
        print(f"Error: Missing required subdirectories: {missing_categories}")
        return
    
    print(f"Source dataset path: {source_dataset_path}")
    print(f"Destination dataset path: {destination_dataset_path}")
    print("Starting dataset organization...")
    
    try:
        # Split the dataset
        split_dataset(
            source_path=source_dataset_path,
            destination_path=destination_dataset_path,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        print("\n" + "="*50)
        print("Dataset organization completed successfully!")
        print("="*50)
        
        # Print summary
        print("\nFinal directory structure:")
        for split in ['training', 'validation', 'test']:
            for category in ['benign', 'malignant', 'normal']:
                dir_path = os.path.join(destination_dataset_path, split, category)
                if os.path.exists(dir_path):
                    all_files = [f for f in os.listdir(dir_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
                    mask_files = [f for f in all_files if '_mask.' in f.lower()]
                    image_files = [f for f in all_files if '_mask.' not in f.lower()]
                    pairs_count = min(len(mask_files), len(image_files))
                    print(f"{split}/{category}: {pairs_count} pairs ({len(all_files)} total files)")
        
    except Exception as e:
        print(f"Error during dataset organization: {str(e)}")

# Example usage
if __name__ == "__main__":
    # First organize the dataset
    main()
    
    print("\n" + "="*60)
    print("DATASET ORGANIZATION COMPLETE - NOW TESTING PYTORCH DATASET")
    print("="*60)
    