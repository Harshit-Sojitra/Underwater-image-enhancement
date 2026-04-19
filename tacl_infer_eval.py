"""
TACL Inference on UFO-120 Test Set
==================================
Runs the TACL (Twin Adversarial Contrastive Learning) model.

Model: ResnetGenerator (9 blocks)
Weights: D:\CTM-main\TACL\checkpoints\tacl_ufo120\latest_net_G.pth
"""
import sys
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import functools

# Add TACL path to sys.path
TACL_DIR = r"D:\CTM-main\TACL"
sys.path.insert(0, TACL_DIR)

# Import the network definition directly from the repo
try:
    from models.networks import ResnetGenerator
except ImportError as e:
    print(f"Error importing TACL networks: {e}")
    sys.exit(1)

# Add evaluation metric path
sys.path.insert(0, r"D:\CTM-main\evaluation matrix")
import uqim_utils

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INPUT_DIR = r"D:\CTM-main\dataset\UFO-120\TEST\lrd"
CHECKPOINT_PATH = r"D:\CTM-main\TACL\checkpoints\tacl_ufo120\latest_net_G.pth"
OUTPUT_DIR = r"D:\CTM-main\results\TACL"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_generator():
    """Manually instantiate and load the TACL ResnetGenerator."""
    # Settings from TACL paper/Readme: resnet_9blocks, instance norm, no dropout
    norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
    
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle DataParallel prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict
        
    netG.load_state_dict(state_dict)
    netG.to(DEVICE)
    netG.eval()
    return netG

def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = load_generator()
    
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png")))
    print(f"Processing {len(files)} images...")
    
    with torch.no_grad():
        for fpath in tqdm(files):
            fname = os.path.basename(fpath)
            
            # Load and preprocess
            img_pil = Image.open(fpath).convert("RGB").resize((256, 256), Image.LANCZOS)
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            
            # Normalize to [-1, 1] as expected by GAN Tanh output
            img_np = (img_np - 0.5) / 0.5
            
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # Inference
            output = model(img_tensor)
            
            # Post-process: [-1, 1] -> [0, 1]
            out_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = (out_img * 0.5) + 0.5
            out_img = np.clip(out_img, 0, 1)
            
            # Save
            out_pil = Image.fromarray((out_img * 255).astype(np.uint8))
            out_pil.save(os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".png"))

def evaluate():
    print("\nEvaluating TACL outputs...")
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
