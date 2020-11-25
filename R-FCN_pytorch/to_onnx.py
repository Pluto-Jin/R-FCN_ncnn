
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

cfg.merge_from_file('configs/citypersons/e2e_cascade_R_FCN.yaml')
cfg.freeze()

# build model from cfg
model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)

if len(cfg.MODEL.DEVICE_IDS) == 1:
    torch.cuda.set_device(cfg.MODEL.DEVICE_IDS[0])
# model to gpu
model.to(device)
model.eval()
#print(model)

# load pre-trained model
if cfg.MODEL.LOAD_PRETRAINED:
    pretrained_model_path = cfg.OUTPUT_DIR + \
                            '/model_' + '{:07d}'.format(cfg.MODEL.PRETRAINED_ITER) + '.pth'
    model_loaded = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    load_state_dict(model, model_loaded['model'])
    print('Loading pre-trained rpn model from {}'.format(pretrained_model_path))

# build image transform
is_train = False
transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)

#
im_mean = [0.485, 0.456, 0.406]
im_std  = [0.229, 0.224, 0.225]

test_image_folder = 'test_images/'


output_onnx = 'RFCN11.onnx'
im = Image.open(test_image_folder+"1.jpg")
im = F.to_tensor(im)
im = F.normalize(im, mean=im_mean, std=im_std)
im = torch.unsqueeze(im,0)
print("image shape",im.size)
x = im.to(device)
#with torch.no_grad():
#    torch_out = model(x)

#print(torch_out)
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input"]
#output_names = ["score","label","bbox"]
output_names = ["score","reg","psf","psf4"]
torch.onnx._export(model, x, output_onnx, export_params=True, verbose=False, opset_version=11, input_names=input_names, output_names=output_names,keep_initializers_as_inputs=True,dynamic_axes={'input':{2:'width',3:'height'}})
#,'score':{0:'len'},'label':{0,'len'},'bbox':{0,'len'}})

