#!/usr/bin/env python3
"""
Calculate Dice scores between predictions and ground truth with proper file matching
"""
import numpy as np
import os
import re
from pathlib import Path
import argparse
from skimage import io, measure, filters
from skimage.transform import resize
import json
import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Convert to binary if not already
    y_true_f = (y_true_f > 0).astype('float32')
    y_pred_f = (y_pred_f > 0).astype('float32')
    
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    return dice

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Convert to binary if not already
    y_true_f = (y_true_f > 0).astype('float32')
    y_pred_f = (y_pred_f > 0).astype('float32')
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def extract_number_from_filename(filename):
    # Remove extension
    name = Path(filename).stem
    
    # Try to match pattern like pred_mask_0000
    match = re.search(r'pred_mask_(\d+)', name)
    if match:
        return match.group(1)
    
    # Try to match pattern like test0000_0000 or test0000
    match = re.search(r'test(\d+)', name)
    if match:
        return match.group(1)
    
    # Try to match pattern like 0000_0000 or just 0000
    match = re.search(r'(\d+)', name)
    if match:
        return match.group(1)
    
    return None

def smooth_contour(contour, smooth_factor=3):
    if len(contour) < smooth_factor * 2:
        return contour
    
    # Pad the contour to handle circular nature
    padded = np.concatenate([contour[-smooth_factor:], contour, contour[:smooth_factor]])
    
    # Apply moving average
    smoothed = np.array([np.mean(padded[i:i+smooth_factor*2+1], axis=0) 
                        for i in range(len(contour))])
    
    return smoothed

def plot_and_save_comparison(pred_img, gt_img, orig_img, pred_filename, gt_filename, dice_score, iou_score, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    pred_binary = (pred_img > 0).astype('uint8')
    gt_binary = (gt_img > 0).astype('uint8')
    
    pred_smoothed = filters.gaussian(pred_binary.astype(float), sigma=1.0)
    gt_smoothed = filters.gaussian(gt_binary.astype(float), sigma=1.0)
    
    if orig_img.ndim == 2:
        # Grayscale image
        ax.imshow(orig_img, cmap='gray')
    elif orig_img.ndim == 3:
        # Color image
        if orig_img.shape[2] == 1:
            ax.imshow(orig_img[:, :, 0], cmap='gray')
        else:
            ax.imshow(orig_img)
    else:
        ax.imshow(orig_img, cmap='gray')
    
    try:
        pred_contours = measure.find_contours(pred_smoothed, level=0.3)
        for contour in pred_contours:
            # Smooth the contour
            smoothed_contour = smooth_contour(contour)
            ax.plot(smoothed_contour[:, 1], smoothed_contour[:, 0], color='yellow', linewidth=4, label='Prediction' if contour is pred_contours[0] else "")
    except Exception as e:
        print(f"Error finding prediction contours: {e}")
    
    try:
        gt_contours = measure.find_contours(gt_smoothed, level=0.3)
        for contour in gt_contours:
            # Smooth the contour
            smoothed_contour = smooth_contour(contour)
            ax.plot(smoothed_contour[:, 1], smoothed_contour[:, 0], color='lime', linewidth=4, label='Ground Truth' if contour is gt_contours[0] else "")
    except Exception as e:
        print(f"Error finding ground truth contours: {e}")
    
    ax.set_title(f'Prediction vs Ground Truth Contours\nDice: {dice_score:.4f}, IoU: {iou_score:.4f}', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    pred_number = extract_number_from_filename(pred_filename)
    output_filename = f'contour_comparison_{pred_number}_{pred_filename.replace(".png", "")}.png'
    output_path = Path(output_dir) / output_filename
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    return str(output_path)

def calculate_dice_scores(pred_folder, gt_folder, output_file=None, plot_results=False, plot_output_dir=None, orig_images_folder=None):
    
    pred_path = Path(pred_folder)
    gt_path = Path(gt_folder)
    
    if plot_results and plot_output_dir is None:
        plot_output_dir = pred_path.parent / 'visualizations'
    
    # Check if original images folder is provided when plotting is requested
    if plot_results and orig_images_folder is None:
        print("Warning: Original images folder not provided. Plotting will be disabled.")
        plot_results = False
    
    orig_path = None
    if plot_results and orig_images_folder is not None:
        orig_path = Path(orig_images_folder)
        if not orig_path.exists():
            print(f"Warning: Original images folder does not exist: {orig_images_folder}. Plotting will be disabled.")
            plot_results = False

    # Get all prediction files
    pred_files = list(pred_path.glob("*.png"))
    
    if not pred_files:
        print(f"No PNG files found in prediction folder: {pred_folder}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    dice_scores = []
    iou_scores = []
    matched_pairs = []
    unmatched_predictions = []
    saved_plots = []
    
    for pred_file in pred_files:
        # Extract number from prediction filename
        pred_number = extract_number_from_filename(pred_file.name)
        
        if pred_number is None:
            print(f"Could not extract number from prediction file: {pred_file.name}")
            unmatched_predictions.append(pred_file.name)
            continue
        
        gt_filename = f"{pred_number}.jpg"
        gt_file = gt_path / gt_filename
        
        if not gt_file.exists():
            gt_filename = f"{pred_number}.png"
            gt_file = gt_path / gt_filename
        
        if not gt_file.exists():
            gt_filename = pred_file.name
            gt_file = gt_path / gt_filename
        
        if not gt_file.exists():
            print(f"Ground truth file not found for {pred_file.name}: tried {pred_number}.jpg, {pred_number}.png, and {pred_file.name}")
            unmatched_predictions.append(pred_file.name)
            continue
        
        try:
            pred_img = io.imread(str(pred_file))
            gt_img = io.imread(str(gt_file))

            if pred_img.ndim == 3:
                pred_img = pred_img[:, :, 0]  # Take first channel
            if gt_img.ndim == 3:
                gt_img = gt_img[:, :, 0]  # Take first channel
            
            if pred_img.shape != gt_img.shape:
                pred_img = resize(pred_img, gt_img.shape, anti_aliasing=True)
            
            dice = dice_coefficient(gt_img, pred_img)
            iou = iou_coefficient(gt_img, pred_img)
            dice_scores.append(dice)
            iou_scores.append(iou)
            matched_pairs.append((pred_file.name, gt_filename, dice, iou))
            
            if plot_results and orig_path is not None:
                try:
                    if gt_filename.startswith('mask_'):
                        orig_filename = gt_filename.replace('mask_', 'bus_')
                    else:
                        orig_filename = gt_filename  # Fallback to same filename as ground truth
                    orig_file = orig_path / orig_filename
                    
                    if orig_file.exists():
                        orig_img = io.imread(str(orig_file))
                        
                        # Handle grayscale original images
                        if orig_img.ndim == 3 and orig_img.shape[2] > 1:
                            orig_img = orig_img[:, :, 0]  # Take first channel if multi-channel
                        
                        plot_path = plot_and_save_comparison(
                            pred_img, gt_img, orig_img, pred_file.name, gt_filename, 
                            dice, iou, plot_output_dir
                        )
                        saved_plots.append(plot_path)
                        print(f"Saved plot: {plot_path}")
                    else:
                        print(f"Original image not found for {pred_file.name} (tried {orig_filename})")
                        
                except Exception as plot_error:
                    print(f"Error saving plot for {pred_file.name}: {plot_error}")
            
        except Exception as e:
            print(f"Error processing {pred_file.name}: {e}")
            unmatched_predictions.append(pred_file.name)
    
    # Calculate statistics
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        median_dice = np.median(dice_scores)
        min_dice = np.min(dice_scores)
        max_dice = np.max(dice_scores)
        
        mean_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)
        median_iou = np.median(iou_scores)
        min_iou = np.min(iou_scores)
        max_iou = np.max(iou_scores)
        
        print(f"\n=== SEGMENTATION RESULTS ===")
        print(f"Total matched pairs: {len(matched_pairs)}")
        print(f"\nDice Score Statistics:")
        print(f"  Mean Dice: {mean_dice:.4f}")
        print(f"  Std Dice: {std_dice:.4f}")
        print(f"  Median Dice: {median_dice:.4f}")
        print(f"  Min Dice: {min_dice:.4f}")
        print(f"  Max Dice: {max_dice:.4f}")
        print(f"\nIoU Score Statistics:")
        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Std IoU: {std_iou:.4f}")
        print(f"  Median IoU: {median_iou:.4f}")
        print(f"  Min IoU: {min_iou:.4f}")
        print(f"  Max IoU: {max_iou:.4f}")
        
        # Save results
        results = {
            "total_pairs": len(matched_pairs),
            "dice_statistics": {
                "mean_dice": float(mean_dice),
                "std_dice": float(std_dice),
                "median_dice": float(median_dice),
                "min_dice": float(min_dice),
                "max_dice": float(max_dice)
            },
            "iou_statistics": {
                "mean_iou": float(mean_iou),
                "std_iou": float(std_iou),
                "median_iou": float(median_iou),
                "min_iou": float(min_iou),
                "max_iou": float(max_iou)
            },
            "individual_scores": [
                {"prediction": pred, "ground_truth": gt, "dice": float(dice), "iou": float(iou)}
                for pred, gt, dice, iou in matched_pairs
            ],
            "unmatched_predictions": unmatched_predictions,
            "plotting_info": {
                "plots_generated": plot_results,
                "total_plots_saved": len(saved_plots) if plot_results else 0,
                "plot_output_directory": str(plot_output_dir) if plot_results else None,
                "saved_plot_paths": saved_plots if plot_results else []
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        # Print plotting info
        if plot_results:
            print(f"\nVisualization plots:")
            print(f"  Total plots saved: {len(saved_plots)}")
            print(f"  Plot output directory: {plot_output_dir}")

        # Print some individual results
        print(f"\nFirst 10 individual results:")
        for i, (pred, gt, dice, iou) in enumerate(matched_pairs[:10]):
            print(f"  {pred} -> {gt}: Dice={dice:.4f}, IoU={iou:.4f}")
        
        if unmatched_predictions:
            print(f"\nUnmatched predictions ({len(unmatched_predictions)}):")
            for pred in unmatched_predictions[:10]:
                print(f"  {pred}")
            if len(unmatched_predictions) > 10:
                print(f"  ... and {len(unmatched_predictions) - 10} more")
    
    else:
        print("No valid dice or IoU scores calculated!")

def main():
    parser = argparse.ArgumentParser(description='Calculate Dice and IoU scores between predictions and ground truth')
    parser.add_argument('pred_folder', help='Path to prediction folder')
    parser.add_argument('gt_folder', help='Path to ground truth folder')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)', default=None)
    parser.add_argument('--plot', action='store_true', help='Generate and save visualization plots')
    parser.add_argument('--plot-dir', help='Directory to save plots (default: visualizations/ in parent of pred_folder)', default=None)
    parser.add_argument('--orig-images', help='Path to original test images folder (required for plotting)', default=None)
    
    args = parser.parse_args()
    
    calculate_dice_scores(args.pred_folder, args.gt_folder, args.output, args.plot, args.plot_dir, args.orig_images)

if __name__ == "__main__":
    main() 
    
    