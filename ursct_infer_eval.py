"""
URSCT Inference on UFO-120 Test Set
====================================
Runs ALL 3 pretrained URSCT_SR checkpoints on the UFO-120 LRD test images.
Uses the reference evaluation protocol: PIL load + resize to 256x256.

Checkpoints:
  - UFO_SRx2  (trained on UFO-120, most relevant to paper)
  - LSUI      (trained on LSUI dataset)
  - UIEB      (trained on UIEB dataset)
"""
import sys
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add URSCT model path
URSCT_DIR = r"D:\CTM-main\URSCT-SESR"
sys.path.insert(0, URSCT_DIR)
from model.URSCT_SR_model import URSCT_SR

# Add evaluation metric path
sys.path.insert(0, r"D:\CTM-main\evaluation matrix")
import uqim_utils

# ──────────────────────────────────────────────────────────────
# MODEL CONFIG (from Enh&SR_opt.yaml — UFO SRx2 settings)
# ──────────────────────────────────────────────────────────────
MODEL_OPT = {
    'IN_CHANS'        : 3,
    'OUT_CHANS'       : 3,
    'HR_SIZE'         : [256, 256],   # HR output size
    'SCALE'           : 2,            # UFO_SRx2
    'PATCH_SIZE'      : 2,
    'WIN_SIZE'        : 8,
    'EMB_DIM'         : 32,
    'DEPTH_EN'        : [8, 8, 8, 8],
    'HEAD_NUM'        : [8, 8, 8, 8],
    'MLP_RATIO'       : 4.0,
    'QKV_BIAS'        : True,
    'QK_SCALE'        : 8,
    'DROP_RATE'       : 0,
    'ATTN_DROP_RATE'  : 0.,
    'DROP_PATH_RATE'  : 0.1,
    'APE'             : False,
    'PATCH_NORM'      : True,
    'USE_CHECKPOINTS' : False,
}

# Input size = HR_SIZE / SCALE = 256/2 = 128x128
LR_INPUT_SIZE = MODEL_OPT['HR_SIZE'][0] // MODEL_OPT['SCALE']  # 128

# UFO-120 test images (low-resolution degraded)
INPUT_DIR = r"D:\CTM-main\dataset\UFO-120\TEST\lrd"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CHECKPOINTS = {
    "URSCT_UFO_SRx2": r"D:\CTM-main\checkpoints\model_URSCT\models\UFO_SRx2\model_bestSSIM.pth",
    "URSCT_LSUI"    : r"D:\CTM-main\checkpoints\model_URSCT\models\LSUI\model_bestSSIM.pth",
    "URSCT_UIEB"    : r"D:\CTM-main\checkpoints\model_URSCT\models\UIEB\model_bestSSIM.pth",
}


def load_model(ckpt_path, device):
    model = URSCT_SR(MODEL_OPT).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Handle different checkpoint formats
    if 'state_dict' in ckpt:
        state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt):
        state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    else:
        state = ckpt
        
    # Remove attn_mask keys as their shape depends on input resolution and can differ
    keys_to_delete = [k for k in state.keys() if 'attn_mask' in k]
    for k in keys_to_delete:
        del state[k]
        
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_inference(model, input_dir, output_dir, device):
    """Run URSCT on all images, save results."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.png"))
    )
    print(f"  Processing {len(files)} images → {output_dir}")
    with torch.no_grad():
        for fpath in tqdm(files, desc="  Inference"):
            fname = os.path.basename(fpath)
            # Load image with PIL, resize to LR input size (128x128)
            img = Image.open(fpath).convert("RGB").resize(
                (LR_INPUT_SIZE, LR_INPUT_SIZE), Image.LANCZOS
            )
            inp = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            inp = inp.to(device)

            out = model(inp)  # Output is HR (256x256)
            out = torch.clamp(out, 0, 1)

            # Convert to PIL and save
            out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_pil = Image.fromarray(out_np, 'RGB')
            # Save as PNG to preserve quality
            save_name = os.path.splitext(fname)[0] + ".png"
            out_pil.save(os.path.join(output_dir, save_name))
    print(f"  Saved {len(files)} images.")


def eval_folder_uiqm(folder_path, resize=(256, 256)):
    """Exact reference-standard evaluation: PIL load + resize to 256x256."""
    files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )
    if not files:
        return None, None
    uiqm_scores, uicm_scores = [], []
    for f in files:
        im = Image.open(f).convert("RGB").resize(resize, Image.LANCZOS)
        arr = np.array(im)
        uiqm  = uqim_utils.getUIQM(arr)
        uicm  = uqim_utils._uicm(arr.astype(np.float32))
        uiqm_scores.append(uiqm)
        uicm_scores.append(uicm)
    return float(np.mean(uiqm_scores)), float(np.mean(uicm_scores))


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        output_dir = os.path.join(r"D:\CTM-main\results", name)
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {name}")
        print(f"  Weights   : {ckpt_path}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        # Load model
        print("  Loading model...")
        model = load_model(ckpt_path, DEVICE)
        print("  Model loaded successfully.")

        # Run inference
        run_inference(model, INPUT_DIR, output_dir, DEVICE)

        # Evaluate
        avg_uiqm, avg_uicm = eval_folder_uiqm(output_dir)
        results[name] = (avg_uicm, avg_uiqm)
        print(f"\n  UIQM = {avg_uiqm:.4f}")
        print(f"  UICM = {avg_uicm:.4f}")

        # Free VRAM between models
        del model
        torch.cuda.empty_cache()

    # ── Also re-evaluate CTM and Water-Net for complete table ──
    other_dirs = {
        "CTM (Fine-tuned)"  : r"D:\CTM-main\results\UFO120_enhanced",
        "Water-Net"         : r"D:\CTM-main\Water-Net_Code\sample",
    }
    for name, folder in other_dirs.items():
        if os.path.exists(folder):
            avg_uiqm, avg_uicm = eval_folder_uiqm(folder)
            results[name] = (avg_uicm, avg_uiqm)

    # ── Final Comparison Table ─────────────────────────────────
    print(f"\n\n{'#'*65}")
    print(f"  COMPLETE COMPARISON TABLE — UFO-120 Test Set")
    print(f"  Evaluation: PIL resize to 256x256 (reference standard)")
    print(f"{'#'*65}")
    print(f"  {'Model':<22}  {'UICM':>12}  {'UIQM':>10}")
    print(f"  {'-'*50}")
    for model, (uicm, uiqm) in results.items():
        print(f"  {model:<22}  {uicm:>12.4f}  {uiqm:>10.4f}")

    print(f"\n  {'--- Paper Reference (UFO-120) ---':}")
    paper_vals = {
        "Water-Net (Paper)"  : (-21.887, 4.2835),
        "URSCT (Paper)"      : (-26.398, 4.0611),
        "CTM/Ours (Paper)"   : ( 3.4094, 4.8883),
    }
    for model, (uicm, uiqm) in paper_vals.items():
        print(f"  {model:<22}  {uicm:>12.4f}  {uiqm:>10.4f}")
    print(f"{'#'*65}\n")
