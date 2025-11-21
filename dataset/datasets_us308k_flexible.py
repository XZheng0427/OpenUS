import os
import math
import random
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image as PILImage
from torchvision.transforms.functional import to_pil_image


class FlexibleUltrasoundImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        patch_size: int,
        pred_ratio: float,
        pred_ratio_var: float,
        pred_aspect_ratio: Tuple[float, float],
        use_sets: Optional[List[str]] = None, 
        img_extensions: List[str] = [".png", ".jpg", ".jpeg", ".tiff", ".tif"],
        transform: Optional[Callable] = None,
        pred_shape: str = 'block',
        pred_start_epoch: int = 0,
        flexible_structure: bool = True  # New parameter for flexible loading
    ):
        self.root = Path(root).expanduser()
        self.transform = transform
        self.samples: List[Path] = []
        self.flexible_structure = flexible_structure

        self.psz = patch_size
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.log_aspect_ratio = tuple(math.log(x) for x in pred_aspect_ratio)
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch    

        # which dataset sub-folders to walk through?
        dataset_dirs = (
            [self.root / d for d in use_sets]
            if use_sets is not None
            else [p for p in self.root.iterdir() if p.is_dir()]
        )

        print(f"Found {len(dataset_dirs)} dataset directories")
        
        for ds in sorted(dataset_dirs):
            if self.flexible_structure:
                # More flexible approach - try multiple strategies
                images_found = self._collect_images_flexible(ds, img_extensions)
            else:
                # Original strict approach
                images_found = self._collect_images_strict(ds, img_extensions[0])
            
            print(f"  {ds.name}: {len(images_found)} images")
            self.samples.extend(images_found)

        print(f"Total images loaded: {len(self.samples)}")
        
        if not self.samples:
            raise RuntimeError("No images found under the specified root!")

    def _collect_images_strict(self, ds_dir: Path, img_suffix: str) -> List[Path]:
        """Original strict collection method"""
        img_dir = ds_dir / "images"
        if not img_dir.is_dir():
            return []
        return sorted(img_dir.glob(f"*{img_suffix}"))
    
    def _collect_images_flexible(self, ds_dir: Path, img_extensions: List[str]) -> List[Path]:
        """Flexible collection method that tries multiple strategies"""
        images = []
        
        img_dir = ds_dir / "images"
        if img_dir.is_dir():
            for ext in img_extensions:
                images.extend(sorted(img_dir.glob(f"*{ext}")))
        
        if not images:  # Only if no images found in images/ subdirectory
            for ext in img_extensions:
                images.extend(sorted(ds_dir.glob(f"*{ext}")))
        
        if not images:  # Only if still no images found
            for subdir in ds_dir.iterdir():
                if subdir.is_dir():
                    for ext in img_extensions:
                        found_in_subdir = list(subdir.glob(f"*{ext}"))
                        if found_in_subdir:
                            images.extend(sorted(found_in_subdir))
                            break  # Found images in this subdir, move to next
        
        return images

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0
        if isinstance(self.pred_ratio, list):
            ratios = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv, "pred_ratio must be at least as large as pred_ratio_var"
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                ratios.append(pr)
            pred_ratio = random.choice(ratios)
        else:
            assert self.pred_ratio >= self.pred_ratio_var, "pred_ratio must be at least as large as pred_ratio_var"
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        image = read_image(str(img_path))
        
        image = to_pil_image(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        masks = []
        
        for img in image:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count
                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)
                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if not mask[i, j]:
                                            mask[i, j] = True
                                            delta += 1
                        if delta > 0:
                            break
                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high))
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
            else:
                raise ValueError("Unsupported pred_shape: {}".format(self.pred_shape))
            
            masks.append(mask)
 
        return image, 0, masks

    def calculate_mean_std(self, num_samples: Optional[int] = None, batch_size: int = 32) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        print("Calculating dataset mean and standard deviation...")
        
        original_transform = self.transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        total_samples = len(self.samples)
        if num_samples is None:
            num_samples = total_samples
        else:
            num_samples = min(num_samples, total_samples)
        
        if num_samples < total_samples:
            indices = random.sample(range(total_samples), num_samples)
        else:
            indices = list(range(total_samples))
        
        mean_sum = torch.zeros(3)
        std_sum = torch.zeros(3)
        total_pixels = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            
            for idx in batch_indices:
                img_path = self.samples[idx]
                image = read_image(str(img_path))
                image = to_pil_image(image)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image = self.transform(image)
                batch_images.append(image)
            
            if batch_images:
                batch_tensor = torch.stack(batch_images)
                batch_mean = batch_tensor.mean(dim=[0, 2, 3])
                mean_sum += batch_mean * batch_tensor.size(0)
                batch_std = batch_tensor.std(dim=[0, 2, 3])
                std_sum += batch_std * batch_tensor.size(0)
                total_pixels += batch_tensor.size(0)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i + batch_size, len(indices))}/{len(indices)} samples")
        
        if total_pixels > 0:
            final_mean = mean_sum / total_pixels
            final_std = std_sum / total_pixels
        else:
            final_mean = torch.zeros(3)
            final_std = torch.ones(3)
        
        self.transform = original_transform
        
        mean_tuple = tuple(final_mean.tolist())
        std_tuple = tuple(final_std.tolist())
        
        print(f"Dataset statistics calculated from {total_pixels} samples:")
        print(f"Mean (R, G, B): {mean_tuple}")
        print(f"Std (R, G, B): {std_tuple}")
        
        return mean_tuple, std_tuple


if __name__ == "__main__":
    root = "."

    # Test with flexible loading
    print("=== Testing Flexible Dataset Loading ===")
    ds_flexible = FlexibleUltrasoundImageDataset(
        root, 
        transform=None,
        patch_size=4, 
        pred_shape='block', 
        pred_ratio=0.3, 
        pred_ratio_var=0, 
        pred_aspect_ratio=(0.3, 1/0.3),
        flexible_structure=True  # Enable flexible loading
    )
    
    print(f"\nFlexible loading found: {len(ds_flexible)} images")
    
    mean, std = ds_flexible.calculate_mean_std(batch_size=16) 
    print(f"Using normalization - Mean: {mean}, Std: {std}")