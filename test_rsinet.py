import argparse
import torch
import random
import os
import cv2
import warnings
import torch.nn as nn
import numpy as np
import pytorch_msssim as torchssim
from tqdm import tqdm
from torch.backends import cudnn
import matplotlib.pyplot as plt

import fjn_util
from model import net

task_scale = {'color': 1, 'colorx2': 2, 'srx2': 2, 'srx4': 4, 'srx8': 8}
scale = 2

task = 'srx2'
device = 'cpu'
checkpoint = '/workspaces/prototype/onnx_2x/x2.pkl'
model = net.Kong(scale=scale).to(device).eval()
pkl = torch.load(checkpoint, map_location=torch.device('cpu'))
state = pkl["model_state_dict"]
model.load_state_dict(state)
print(f'PSNR: {pkl["best_psnr"]}')
print(f'SSIM: {pkl["best_ssim"]}')


import onnxruntime as ort
import numpy as np
import cv2

def perform_inference_onnx(input_tensor):
    print("starting inference")
    ort_session = ort.InferenceSession("model.onnx")
    print("onnx model loaded")
    input_data = input_tensor.numpy()

    inputs = {ort_session.get_inputs()[0].name: input_data}
    print("actual inference")
    out_bgr_hr = ort_session.run(None, inputs)[0]
    print("inference done")

    out_bgr_hr_cpu = np.clip(out_bgr_hr[0] * 1, 0, 255).transpose((1, 2, 0)).astype(np.uint8)
    out_rgb_hr_cpu = cv2.cvtColor(out_bgr_hr_cpu, cv2.COLOR_BGR2RGB)

    return out_rgb_hr_cpu

img_bgr_hr = cv2.imread("path", 1)
img_bgr_hr = cv2.resize(img_bgr_hr, (1024, 1024))
img_bgr_lr = cv2.resize(img_bgr_hr,
                        (img_bgr_hr.shape[1] // self.sr_factor, img_bgr_hr.shape[0] // self.sr_factor))

img_bgr_lr, img_bgr_hr = img_bgr_lr.transpose((2, 0, 1)), img_bgr_hr.transpose((2, 0, 1))

input = torch.Tensor(img_bgr_lr.float())
label = torch.Tensor(img_bgr_hr.float())

out_bgr_hr_cpu = perform_inference_onnx(input)

label_cpu = label.data.numpy()
label_cpu = np.clip(label_cpu [0] * 1, 0, 255).transpose((1, 2, 0)).astype(np.uint8)
label_cpu = cv2.cvtColor(label_cpu , cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(cv2.cvtColor(label_cpu, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Output")
plt.imshow(out_bgr_hr_cpu)
plt.axis('off')

plt.show()