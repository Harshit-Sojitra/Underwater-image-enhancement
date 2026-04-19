"""
Batch Test + Evaluate ALL images using CTM Model.
Processes every image in input/ folder, saves enhanced outputs,
computes PSNR, SSIM, LPIPS, FSIM, UICM, UIQM for each pair,
then prints averages at the end.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import piq
from thop import profile as thop_profile

# Add evaluation matrix folder to path for uqim_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evaluation matrix'))
import uqim_utils

from Networks.model import Net

# ============================================================
# CONFIG — Edit these paths if needed
# ============================================================
INPUT_DIR      = r"dataset\UFO-120\TEST\lrd"     # UFO-120 test inputs (degraded)
TARGET_DIR     = r"dataset\UFO-120\TEST\hr"      # UFO-120 test targets (ground truth)
OUTPUT_DIR     = r"results\UFO120_enhanced"       # Output folder
# CHECKPOINT     = r"checkpoints\model_C\model_best.pth"
CHECKPOINT     = r"checkpoints\model_UFO\model_best.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading model...")

# 1. Create the empty brain
model = Net().to(DEVICE)

# 2. Load the memories from the hard drive
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

# 3. Inject the memories into the brain
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded from: {CHECKPOINT}")

# ------ FLOPs and Parameter Count ------
dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
with torch.no_grad():
    flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
flops_G = flops / 1e9   # Convert to GFLOPs
params_M = params / 1e6  # Convert to Millions
print(f"\nModel Complexity:")
print(f"  GFLOPs : {flops_G:.2f} G")
print(f"  Params : {params_M:.2f} M\n")

# Load LPIPS (only once for all images - faster)
print("Loading LPIPS network...")
loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(DEVICE)

# Collect all input images
input_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])
total = len(input_files)
print(f"\nFound {total} images to process.\n")

# Accumulators for averages
metrics = {
    'psnr': [], 'ssim': [], 'lpips': [],
    'fsim': [], 'uicm': [], 'uiqm': []
}
skipped = 0

print(f"{'#':<6} {'Filename':<28} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'FSIM':>7} {'UICM':>8} {'UIQM':>8}")
print("-" * 85)

with torch.no_grad():
    for idx, fname in enumerate(tqdm(input_files, desc="Processing", unit="img")):

        # ------ Input Image ------
        inp_path = os.path.join(INPUT_DIR, fname)
        img = Image.open(inp_path).convert("RGB")
        input_ = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
        orig_h, orig_w = input_.shape[2], input_.shape[3]

        # ------ Check matching target ------
        # Try same filename (jpg or png)
        base_name = os.path.splitext(fname)[0]
        gt_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(TARGET_DIR, base_name + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            skipped += 1
            continue  # No ground truth found, skip

        # ------ Run Model ------
        # Resize to 256x256 square (required by CTM window attention)
        I256 = F.interpolate(input_, size=(256, 256), mode='bilinear', align_corners=False)
        # Returns [out_64, out_128, out_256]
        outputs = model(I256)
        out_256 = outputs[2]  # Full-scale output (256x256)

        # ====== FIX (Problems 2 & 3): Compute UIQM on the raw 256x256 output ======
        # (Before we apply bilinear resize-back which blurs edges and ruins UISM/UIQM)
        out_256_clamped = torch.clamp(out_256, 0, 1)
        out_256_np = out_256_clamped[0].permute(1, 2, 0).cpu().numpy()
        out_256_ubyte = img_as_ubyte(out_256_np)
        enh_float_256 = out_256_ubyte.astype(np.float32)
        val_uicm  = uqim_utils._uicm(enh_float_256)
        val_uiqm  = uqim_utils.getUIQM(enh_float_256)

        # ------ Resize back to original and save ------
        result = out_256_clamped
        result = F.interpolate(result, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        result_np = result[0].permute(1, 2, 0).cpu().numpy()
        result_img = img_as_ubyte(result_np)

        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_enhanced.png")
        cv2.imwrite(out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

        # ------ Load GT for metrics ------
        gt_img = cv2.imread(gt_path)
        gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Make sure sizes match (should already match now)
        if result_img.shape != gt_img_rgb.shape:
            result_img = cv2.resize(result_img, (gt_img_rgb.shape[1], gt_img_rgb.shape[0]))

        # PSNR + SSIM
        val_psnr = psnr(gt_img_rgb, result_img, data_range=255)
        val_ssim = ssim(gt_img_rgb, result_img, channel_axis=2, data_range=255)

        # Tensors for LPIPS / FSIM (0-1 range, BCHW)
        enh_t = torch.from_numpy(result_img.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0
        gt_t  = torch.from_numpy(gt_img_rgb.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0

        val_lpips = loss_fn_vgg(enh_t * 2 - 1, gt_t * 2 - 1).item()
        val_fsim  = piq.fsim(enh_t, gt_t, data_range=1.0).item()

        # Accumulate
        metrics['psnr'].append(val_psnr)
        metrics['ssim'].append(val_ssim)
        metrics['lpips'].append(val_lpips)
        metrics['fsim'].append(val_fsim)
        metrics['uicm'].append(val_uicm)
        metrics['uiqm'].append(val_uiqm)

        # Print per-image result
        tqdm.write(
            f"{idx+1:<6} {fname:<28} "
            f"{val_psnr:>7.4f} {val_ssim:>7.4f} {val_lpips:>7.4f} "
            f"{val_fsim:>7.4f} {val_uicm:>8.4f} {val_uiqm:>8.4f}"
        )

# ============================================================
# FINAL AVERAGE RESULTS
# ============================================================
n = len(metrics['psnr'])
print("\n" + "=" * 85)
print(f"BATCH EVALUATION COMPLETE — {n} images processed, {skipped} skipped (no GT)")
print("=" * 85)
print(f"{'Metric':<12} {'Average':>10}   {'What it measures'}")
print("-" * 85)
print(f"{'PSNR':<12} {np.mean(metrics['psnr']):>10.4f}   Pixel accuracy (dB)       — Higher is better (>20 good)")
print(f"{'SSIM':<12} {np.mean(metrics['ssim']):>10.4f}   Structural similarity     — Higher is better (>0.8 good)")
print(f"{'LPIPS':<12} {np.mean(metrics['lpips']):>10.4f}   Perceptual diff (VGG)     — Lower is better  (<0.2 good)")
print(f"{'FSIM':<12} {np.mean(metrics['fsim']):>10.4f}   Feature similarity        — Higher is better (>0.85 good)")
print(f"{'UICM':<12} {np.mean(metrics['uicm']):>10.4f}   Colorfulness (UW-specific)— Higher is better")
print(f"{'UIQM':<12} {np.mean(metrics['uiqm']):>10.4f}   UW Quality (Color+Sharp)  — Higher is better")
print("=" * 85)

# Model complexity in final summary
print(f"{'GFLOPs':<12} {flops_G:>10.2f}   Computational cost (lower = faster inference)")
print(f"{'Params (M)':<12} {params_M:>10.2f}   Total trainable parameters (millions)")
print("=" * 85)

# Save results to a log file
log_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
with open(log_path, "w") as f:
    f.write(f"CTM Batch Evaluation — {n} images\n")
    f.write("=" * 60 + "\n")
    f.write(f"PSNR      : {np.mean(metrics['psnr']):.4f}\n")
    f.write(f"SSIM      : {np.mean(metrics['ssim']):.4f}\n")
    f.write(f"LPIPS     : {np.mean(metrics['lpips']):.4f}\n")
    f.write(f"FSIM      : {np.mean(metrics['fsim']):.4f}\n")
    f.write(f"UICM      : {np.mean(metrics['uicm']):.4f}\n")
    f.write(f"UIQM      : {np.mean(metrics['uiqm']):.4f}\n")
    f.write(f"GFLOPs    : {flops_G:.2f}\n")
    f.write(f"Params (M): {params_M:.2f}\n")
print(f"\nResults also saved to: {log_path}")
