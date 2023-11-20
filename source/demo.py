"""
Usage:

cd source
0 mse = 1 tv = 500
1 mse = 1 tv = 500 sm = 1 gamma
2 mse = 1 tv = 500 sm = 10 gamma
3 l1 = 1 tv = 500 cc = 1 smooth = 1
4 l1 = 1 tv = 500 cc = 1 smooth = 1 iteration 1
5 l1 = 1 tv = 500 cc = 1 smooth = 1 iteration 2
6 l1 = 5 tv = 500 cc = 1 smooth = 1 iteration 2
python demo.py --input_dir /home/kate.brillantes/thesis/psenet/data/demo/input \
    --output_dir /home/kate.brillantes/thesis/psenet/data/demo/output \
    --checkpoint /home/kate.brillantes/thesis/psenet/workdirs/afifi/version_19/checkpoints/last.ckpt
"""

import argparse
import glob
import os

import cv2
import torch
import torchvision
from model import UnetTMO


def read_image(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = img / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model.") :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="../pretrained/afifi.pth")
parser.add_argument("--input_dir", type=str, default="samples")
parser.add_argument("--output_dir", type=str, default="output")

args = parser.parse_args()

model = UnetTMO()
state_dict = read_pytorch_lightning_state_dict(torch.load(args.checkpoint))
model.load_state_dict(state_dict)
model.eval()
model.cuda()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_images = glob.glob(os.path.join(args.input_dir, "*"))
for path in input_images:
    print("Process:", path)
    image = read_image(path).cuda()
    with torch.no_grad():
        output, _ = model(image)
    torchvision.utils.save_image(output, path.replace(args.input_dir, args.output_dir))
