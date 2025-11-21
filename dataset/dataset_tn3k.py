import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class TN3KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, json_file, split='train', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(json_file, 'r') as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")
        self.ids = splits[split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id:04d}.jpg")
        msk_path = os.path.join(self.mask_dir,  f"{img_id:04d}.jpg")

        # Load
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, img_id


class TN3KTestDataset(Dataset):
    def __init__(self, test_image_dir, test_mask_dir, transform=None):
        self.test_image_dir = test_image_dir
        self.test_mask_dir = test_mask_dir
        self.transform = transform
        
        # Get all image files from the test image directory
        self.image_files = []
        if os.path.exists(test_image_dir):
            all_files = os.listdir(test_image_dir)
            # Filter for common image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            self.image_files = [f for f in all_files 
                              if os.path.splitext(f.lower())[1] in image_extensions]
            self.image_files.sort()  # Sort for consistent ordering
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {test_image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.test_image_dir, img_filename)
        msk_path = os.path.join(self.test_mask_dir, img_filename)

        # Load images
        image = Image.open(img_path).convert('RGB')
        
        # Check if corresponding mask exists
        if os.path.exists(msk_path):
            mask = Image.open(msk_path).convert('L')
        else:
            # If no mask found, create a dummy mask of zeros
            mask = Image.new('L', image.size, 0)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, img_filename
    
    
    
    
