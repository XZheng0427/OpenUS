import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path

class BUSBRADataset(Dataset):
    def __init__(self, organized_data_dir, fold, split='train', transform=None):
        self.organized_data_dir = Path(organized_data_dir)
        self.fold = fold
        self.split = split
        self.transform = transform
        
        # Construct paths to images and masks directories
        fold_dir = self.organized_data_dir / f"fold_{fold}" / split
        self.images_dir = fold_dir / "images"
        self.masks_dir = fold_dir / "masks"
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # Get list of image files
        self.image_files = sorted([f for f in self.images_dir.glob("*.png")])
        if not self.image_files:
            # Try other extensions
            for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                self.image_files = sorted([f for f in self.images_dir.glob(f"*{ext}")])
                if self.image_files:
                    break
        
        if not self.image_files:
            raise ValueError(f"No image files found in {self.images_dir}")
        
        print(f"Loaded {len(self.image_files)} {split} samples from fold {fold}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file path
        img_path = self.image_files[idx]
        img_filename = img_path.name
        
        # Construct corresponding mask path
        # Convert bus_XXXX-X.png to mask_XXXX-X.png
        mask_filename = img_filename.replace('bus_', 'mask_')
        mask_path = self.masks_dir / mask_filename
        
        # Verify mask file exists
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default to simple tensor conversion
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, mask_filename

class BUSBRADatasetLegacy(Dataset):
    """
    Legacy PyTorch Dataset for BUSBRA segmentation using 5-fold cross-validation CSV.
    This is the original version that works with CSV files.

    Args:
        csv_file (str): Path to the CSV file containing IDs and fold assignments.
        images_dir (str): Directory with input images (e.g., .jpg files named by ID).
        masks_dir (str): Directory with corresponding masks (e.g., .png files named by replacing 'bus' with 'mask').
        fold (int): Zero-based fold index (0 through 4) to use as validation/test.
        train (bool): If True, uses all samples not in the specified fold for training; otherwise uses only samples in the fold.
        transform (callable, optional): A function/transform that takes an image and mask and returns transformed versions.
    """
    def __init__(self, csv_file, images_dir, masks_dir, fold, train=True, transform=None):
        self.df = pd.read_csv(csv_file)
        # filter by fold
        if train:
            self.df = self.df[self.df['kFold'] != (fold + 1)].reset_index(drop=True)
        else:
            self.df = self.df[self.df['kFold'] == (fold + 1)].reset_index(drop=True)

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ID']
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        mask_id = img_id.replace('bus', 'mask')
        mask_path = os.path.join(self.masks_dir, f"{mask_id}.png")

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # default to simple tensor conversion
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# Example usage with organized data:
if __name__ == "__main__":
    # New organized data approach
    organized_data_dir = "./BUSBRA/organized_data"
    fold = 1  # Use fold 1
    
    # Create datasets for train, validation, and test
    train_dataset = BUSBRADataset(organized_data_dir, fold, split='train')
    val_dataset = BUSBRADataset(organized_data_dir, fold, split='validation')
    test_dataset = BUSBRADataset(organized_data_dir, fold, split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"Fold {fold} - Train samples: {len(train_dataset)}")
    print(f"Fold {fold} - Validation samples: {len(val_dataset)}")
    print(f"Fold {fold} - Test samples: {len(test_dataset)}")
    
    # Test loading a sample
    print("\nTesting data loading...")
    train_image, train_mask, train_mask_filename = train_dataset[0]
    print(f"Train image shape: {train_image.shape}")
    print(f"Train mask shape: {train_mask.shape}")
    print(f"Train mask filename: {train_mask_filename}")
    
    # Example for all folds
    print("\n" + "="*50)
    print("ALL FOLDS SUMMARY")
    print("="*50)
    
    for fold_num in range(1, 6):
        try:
            train_ds = BUSBRADataset(organized_data_dir, fold_num, split='train')
            val_ds = BUSBRADataset(organized_data_dir, fold_num, split='validation')
            test_ds = BUSBRADataset(organized_data_dir, fold_num, split='test')
            
            print(f"Fold {fold_num}: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        except Exception as e:
            print(f"Fold {fold_num}: Error - {e}")
    
    # Legacy CSV-based approach (for comparison)
    print("\n" + "="*50)
    print("LEGACY CSV APPROACH (for comparison)")
    print("="*50)
    
    csv_file = "./BUSBRA/5-fold-cv.csv"
    images_dir = "./BUSBRA/Images/"
    masks_dir = "./BUSBRA/Masks/"
    
    # Note: Legacy uses 0-based fold indexing
    legacy_fold = 0  # corresponds to fold 1 in organized data
    
    try:
        legacy_train_dataset = BUSBRADatasetLegacy(csv_file, images_dir, masks_dir, legacy_fold, train=True)
        legacy_val_dataset = BUSBRADatasetLegacy(csv_file, images_dir, masks_dir, legacy_fold, train=False)
        
        print(f"Legacy Fold {legacy_fold+1} - Train samples: {len(legacy_train_dataset)}")
        print(f"Legacy Fold {legacy_fold+1} - Val samples: {len(legacy_val_dataset)}")
    except Exception as e:
        print(f"Legacy approach error: {e}")