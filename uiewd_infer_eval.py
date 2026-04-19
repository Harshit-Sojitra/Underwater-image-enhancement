"""
UIE-WD Inference on UFO-120 Test Set
=====================================
Runs the UIE-WD (Wavelet-based Dual-stream Network) on UFO-120 LRD images.

Model: Dual_cnn
Weights: D:\\CTM-main\\checkpoints\\model_UIEWD\\model.pth
"""
import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2

# Add UIE-WD path to sys.path to import components
UIEWD_DIR = r"D:\CTM-main\UIE-WD_Code"
sys.path.insert(0, UIEWD_DIR)

# Mock/Redefine components if needed to avoid pywt dependency if missing
# However, it's better to let it fail and ask user to install pywt if they haven't.

try:
    from models.networks_multi import Dual_cnn, WavePool
except ImportError as e:
    print(f"Error importing UIEWD models: {e}")
    print("Please ensure you have installed requirements: pip install PyWavelets scikit-image")
    sys.exit(1)

# Add evaluation metric path
sys.path.insert(0, r"D:\CTM-main\evaluation matrix")
import uqim_utils

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INPUT_DIR = r"D:\CTM-main\dataset\UFO-120\TEST\lrd"
CHECKPOINT_PATH = r"D:\CTM-main\checkpoints\model_UIEWD\model.pth"
OUTPUT_DIR = r"D:\CTM-main\results\UIEWD"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_input(img_pil):
    """
    Implementation of UIE-WD preprocessing:
    1. RGB [0,1]
    2. Wavelet Pool -> LL, LH, HL, HH
    3. Structure = LL_RGB + LL_LAB + LL_HSV
    4. Detail = [LH, HL, HH]
    """
    # Resize to standard if needed, but paper uses original or 256x256
    # For UFO-120 test, we should keep it standard or match training.
    # UIE-WD training used 256x256.
    img_pil = img_pil.resize((256, 256), Image.LANCZOS)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    
    # img to tensor (1, 3, H, W)
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    
    # Wavelet decomposition
    with torch.no_grad():
        wavePool = WavePool(3).to(DEVICE)
        LL, LH, HL, HH = wavePool(img_tensor)
    
    # Process LL for structure (RGB + LAB + HSV)
    # LL is (1, 3, H/2, W/2)
    ll_np = LL.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # cv2 color conversions expect [0, 1] or [0, 255]
    # LL output of WavePool is 1/sqrt(2)*sum, so it might be slightly > 1.
    # Clipping is safer for cv2 color conversion
    ll_np_clipped = np.clip(ll_np, 0, 1)
    
    ll_lab = cv2.cvtColor(ll_np_clipped, cv2.COLOR_RGB2LAB) # This is LAB space
    ll_hsv = cv2.cvtColor(ll_np_clipped, cv2.COLOR_RGB2HSV) # This is HSV space
    
    # Convert to tensors
    ll_rgb_t = torch.from_numpy(ll_np).permute(2, 0, 1).to(DEVICE)
    ll_lab_t = torch.from_numpy(ll_lab).permute(2, 0, 1).to(DEVICE)
    ll_hsv_t = torch.from_numpy(ll_hsv).permute(2, 0, 1).to(DEVICE)
    
    structure = torch.cat([ll_rgb_t, ll_lab_t, ll_hsv_t], dim=0).unsqueeze(0)
    detail = torch.cat([LH, HL, HH], dim=1) # (1, 9, H/2, W/2)
    
    return structure, detail

def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading UIE-WD model from {CHECKPOINT_PATH}...")
    model = Dual_cnn().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")
    
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png")))
    print(f"Processing {len(files)} images...")
    
    with torch.no_grad():
        for fpath in tqdm(files):
            fname = os.path.basename(fpath)
            img_pil = Image.open(fpath).convert("RGB")
            
            structure, detail = prepare_input(img_pil)
            
            # structure, detail, output, structure_ori, detail3 = model(structure, detail)
            _, _, output, _, _ = model(structure, detail)
            
            # IMPORTANT: Dual_cnn uses Tanh → output range is [-1, 1]
            # Must convert to [0, 1] BEFORE converting to uint8
            # (same as original test_multi.py's to_img() function)
            out_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = (out_img + 1.0) / 2.0    # [-1,1] → [0,1]
            out_img = np.clip(out_img, 0, 1)
            out_pil = Image.fromarray((out_img * 255).astype(np.uint8))
            
            # Save as PNG
            out_pil.save(os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".png"))

def evaluate():
    print("\nEvaluating UIE-WD outputs...")
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
    if not files:
        print("No output files found to evaluate.")
        return
        
    uiqm_scores, uicm_scores = [], []
    for f in files:
        im = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
        arr = np.array(im)
        uiqm = uqim_utils.getUIQM(arr)
        uicm = uqim_utils._uicm(arr.astype(np.float32))
        uiqm_scores.append(uiqm)
        uicm_scores.append(uicm)
        
    avg_uiqm = np.mean(uiqm_scores)
    avg_uicm = np.mean(uicm_scores)
    
    print(f"Average UIQM: {avg_uiqm:.4f}")
    print(f"Average UICM: {avg_uicm:.4f}")
    
    return avg_uicm, avg_uiqm

if __name__ == "__main__":
    run_inference()
    evaluate()
