import argparse
import os
import cv2
import numpy as np
import torch
import lpips
import piq
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import uqim_utils

def evaluate_image(enhanced_path, gt_path):
    print(f"--- Evaluaton Matrix ---")
    print(f"Enhanced Image: {enhanced_path}")
    print(f"Ground Truth: {gt_path}\n")

    # Load images for skimage (numpy, HWC, RGB)
    enh_img = cv2.imread(enhanced_path)
    gt_img = cv2.imread(gt_path)

    if enh_img is None or gt_img is None:
        print("Error: Could not load one or both images. Please check the paths.")
        return

    # Resize enhanced to match GT if necessary
    if enh_img.shape != gt_img.shape:
        print(f"Warning: Resizing enhanced image {enh_img.shape} to match GT {gt_img.shape}")
        enh_img = cv2.resize(enh_img, (gt_img.shape[1], gt_img.shape[0]))

    enh_img_rgb = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)
    gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    # 1. PSNR & SSIM
    val_psnr = psnr(gt_img_rgb, enh_img_rgb)
    val_ssim = ssim(gt_img_rgb, enh_img_rgb, channel_axis=2, data_range=255)

    print(f"1. PSNR : {val_psnr:.4f}")
    print(f"2. SSIM : {val_ssim:.4f}")

    # Prepare tensors for PyTorch metrics (BCHW, 0-1 range)
    enh_tensor = torch.from_numpy(enh_img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    gt_tensor = torch.from_numpy(gt_img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # 3. LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False)
    # LPIPS expects inputs in [-1, 1] range
    enh_tensor_lpips = enh_tensor * 2 - 1
    gt_tensor_lpips = gt_tensor * 2 - 1
    val_lpips = loss_fn_vgg(enh_tensor_lpips, gt_tensor_lpips).item()
    print(f"3. LPIPS: {val_lpips:.4f}")

    # 4. FSIM (Using piq)
    val_fsim = piq.fsim(enh_tensor, gt_tensor, data_range=1.0).item()
    print(f"4. FSIM : {val_fsim:.4f}")

    # 5 & 6. UICM and UIQM (Reference-free)
    # UIQM requires input as float32 array
    enh_img_rgb_float = enh_img_rgb.astype(np.float32)
    val_uicm = uqim_utils._uicm(enh_img_rgb_float)
    val_uiqm = uqim_utils.getUIQM(enh_img_rgb_float)
    
    print(f"5. UICM : {val_uicm:.4f}")
    print(f"6. UIQM : {val_uiqm:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Enhanced Images")
    parser.add_argument('--enhanced', type=str, required=True, help='Path to enhanced output image')
    parser.add_argument('--gt', type=str, required=True, help='Path to Ground Truth reference image')
    
    args = parser.parse_args()
    evaluate_image(args.enhanced, args.gt)
