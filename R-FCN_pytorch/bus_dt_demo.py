
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import numpy as np
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
#'''

for file in listdir(test_image_folder):
    print()
    im = Image.open(test_image_folder + '1.jpg')
    im_raw = im
    im = F.to_tensor(im)
    im = F.normalize(im, mean=im_mean, std=im_std)
    im = torch.unsqueeze(im,0)
    print("image shape",im.shape)
    im = im.to(device)
    dt = model(im)
    #print(dt)
    break

    # show dt
    #if True:
    if False:
        fig, ax = plt.subplots(1)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.01)
        ax.set_axis_off()
        # [axs.set_axis_off() for axs in ax.ravel()]

        #boxes = dt.convert('xywh').bbox.cpu()
        #labels = dt.get_field('labels')
        #scores = dt.get_field('scores')
        scores,labels,boxes = dt
        
        ax.imshow(im_raw)
        ax.title.set_text('Detection result')

        #print(boxes)
        for idx in range(boxes.shape[0]):
            bb = boxes[idx, :]
            rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(bb[0], bb[1], str('{}'.format(   category2num[str(int(labels[idx]))]   )  ), size=12, color='r')
            ax.text(bb[0] + bb[2], bb[1] + bb[3], str('{:.2f}'.format(scores[idx])), size=12, color='b')
        a = 1

    #fig.savefig("../res_"+file);


print('done')
