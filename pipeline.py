import os
import shutil
import subprocess
import argparse

def setup_test_dataset(raw_dir='./dataset/raw', test_dir='./dataset/raw/test/Cycle_600'):
    """Organizes test images into the required directory structure for test.py"""
    os.makedirs(test_dir, exist_ok=True)
    images_moved = 0
    
    # Find all images in raw_dir
    files = os.listdir(raw_dir)
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(raw_dir, f)
            dst = os.path.join(test_dir, f)
            if not os.path.isdir(src):
                shutil.copy2(src, dst)
                images_moved += 1
                
    print(f"✅ Setup: Copied {images_moved} images to {test_dir}")

def setup_train_dataset(lsui_dir='./dataset/LSUI/LSUI'):
    """Prepares the LSUI dataset by renaming 'GT' to 'target' (expected by dataloader)"""
    gt_dir = os.path.join(lsui_dir, 'GT')
    target_dir = os.path.join(lsui_dir, 'target')
    
    if os.path.exists(gt_dir):
        print(f"Found 'GT' directory in {lsui_dir}. Renaming to 'target' for CTM DataLoader...")
        os.rename(gt_dir, target_dir)
        print("✅ Training dataset directory structure verified.")
    elif os.path.exists(target_dir):
        print("✅ Training dataset 'target' directory already exists.")
    else:
        print(f"⚠️ Warning: Neither 'GT' nor 'target' found in {lsui_dir}.")

def run_inference(gpus='0', size=256):
    """Runs test.py for inference"""
    print(f"\n🚀 Running Inference on GPU {gpus} with input size {size}x{size}...")
    cmd = [
        "python", "test.py", 
        "--input_dir", "./dataset/raw", 
        "--dataset", "Cycle_600", 
        "--weights", "./checkpoints/model_C/model_best.pth", 
        "--gpus", str(gpus),
        "--test_size", str(size)
    ]
    subprocess.run(cmd)

def run_training(gpus='0'):
    """Runs train.py for training"""
    print(f"\n🚀 Starting Training on GPU {gpus}...")
    # Make sure to update training.yml paths beforehand
    cmd = ["python", "train.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTM Pipeline Coordinator")
    parser.add_argument('--step', type=str, choices=['setup_test', 'test', 'setup_train', 'train', 'all_test'], 
                        required=True, help='Which step of the pipeline to run')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID to use')
    
    args = parser.parse_args()
    
    print("-" * 50)
    print(f"CTM Pipeline Runner - Executing: {args.step}")
    print("-" * 50)
    
    if args.step == 'setup_test':
        setup_test_dataset()
    elif args.step == 'test':
        run_inference(gpus=args.gpus)
    elif args.step == 'all_test':
        setup_test_dataset()
        run_inference(gpus=args.gpus)
    elif args.step == 'setup_train':
        setup_train_dataset()
    elif args.step == 'train':
        run_training(gpus=args.gpus)
