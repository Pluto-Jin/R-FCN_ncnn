import onnxruntime

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import onnx
import argparse
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.data.transforms import build_transforms
from torchvision.transforms import functional as F

ort_session = onnxruntime.InferenceSession("RFCN11.onnx")
category2num = {
    'bus': 1,
    'car': 2,
    'van': 3,
    'others': 4,
}

category2num = {
    '1': 'bus',
    '2': 'car',
    '3': 'van',
    '4': 'others',
}


#
im_mean = [0.485, 0.456, 0.406]
im_std  = [0.229, 0.224, 0.225]

test_image_folder = 'test_images/'
#'''

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

print()
im = Image.open(test_image_folder+'1.jpg')
im_raw = im
im = F.to_tensor(im)
im = F.normalize(im, mean=im_mean, std=im_std)
im = torch.unsqueeze(im,0)
print("image shape",im.shape)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(im)}
s,r,f,f4 = ort_session.run(None, ort_inputs)
print()
print("objectness:",s.shape)
print(s)
print("regression:",r.shape)
print(r)
print("psf:",f.shape)
print(f)
print("psf4",f4.shape)
print(f4)



# compute ONNX Runtime output prediction

