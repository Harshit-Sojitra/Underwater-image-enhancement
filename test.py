import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
from thop import profile

from data_RGB import get_test_data
from Networks.model import Net
from skimage import img_as_ubyte

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='model_C', type=str, help='Network Type')
parser.add_argument('--input_dir', default='./dataset/raw', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/model_C/model_latest.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Cycle_600', type=str, help='Test Dataset')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--test_size', default=256, type=int, help='Resize input to this square size for inference')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = Net()
model_restoration.cuda()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ",args.weights)

model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, 'test', dataset)
print(rgb_dir_test)
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

result_dir = os.path.join(args.result_dir, args.network, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_ = data_test[0].cuda()
        filenames = data_test[1]

        # Save original size for restoring output dimensions
        orig_h, orig_w = input_.shape[2], input_.shape[3]

        # Resize to square for CTM (window attention requires square input)
        test_size = args.test_size
        input_256 = F.interpolate(input_, size=(test_size, test_size), mode='bilinear', align_corners=False)
        input_128 = F.interpolate(input_256, scale_factor=0.5)
        input_64 = F.interpolate(input_128, scale_factor=0.5)

        # Save Downsampled Inputs for verification
        utils.save_img(os.path.join(result_dir, filenames[0]+'_input_256.png'), img_as_ubyte(input_256[0].permute(1, 2, 0).cpu().detach().numpy()))
        utils.save_img(os.path.join(result_dir, filenames[0]+'_input_128.png'), img_as_ubyte(input_128[0].permute(1, 2, 0).cpu().detach().numpy()))
        utils.save_img(os.path.join(result_dir, filenames[0]+'_input_64.png'), img_as_ubyte(input_64[0].permute(1, 2, 0).cpu().detach().numpy()))

        restored = model_restoration(input_256)

        # Process and save 3 scales of output
        # Scale 1/4 (64x64)
        restored_64 = torch.clamp(restored[0], 0, 1).permute(0, 2, 3, 1).cpu().detach().numpy()
        utils.save_img(os.path.join(result_dir, filenames[0]+'_output_64.png'), img_as_ubyte(restored_64[0]))

        # Scale 1/2 (128x128)
        restored_128 = torch.clamp(restored[1], 0, 1).permute(0, 2, 3, 1).cpu().detach().numpy()
        utils.save_img(os.path.join(result_dir, filenames[0]+'_output_128.png'), img_as_ubyte(restored_128[0]))

        # Scale 1 (Full size restored)
        restored_256 = torch.clamp(restored[2], 0, 1)
        # Also save the 256x256 version before resizing back to original
        utils.save_img(os.path.join(result_dir, filenames[0]+'_output_256.png'), img_as_ubyte(restored_256[0].permute(1, 2, 0).cpu().detach().numpy()))

        # Resize back to original dimensions for final output
        restored_final = F.interpolate(restored_256, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        restored_final = restored_final.permute(0, 2, 3, 1).cpu().detach().numpy()

        restored_img_final = img_as_ubyte(restored_final[0])
        utils.save_img((os.path.join(result_dir, filenames[0]+'.png')), restored_img_final)

    # Compute FLOPs on the resized input
    flops, params = profile(model_restoration, inputs=(input_256,))
    print('flops: ', flops, 'params: ', params)

print(f"\nDone! {ii+1} images processed. Results saved to: {result_dir}")

















#  give me single step by step code and instruction that to i know what is doing in this and starting on single image and mention all thing which is ongoing and what working process that to i know
