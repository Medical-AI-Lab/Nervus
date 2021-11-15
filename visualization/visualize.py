#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import pandas as pd
import PIL

import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from efficientnet_pytorch import EfficientNet

from gradcam.utils import visualize_cam
from gradcam.gradcam import GradCAM, GradCAMpp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import static
from lib.options import Options
from lib.util import *
from config.mlp_cnn import CreateModel_MLPCNN


args = Options().parse()


train_opt_log_dir = static.train_opt_log_dir
weight_dir = static.weight_dir
df_source = static.df_source
num_classes = static.num_classes
input_list_normed = static.input_list_normed
num_inputs = len(input_list_normed)
visualization_dir = static.visualization_dir
visualization_weight = get_target(weight_dir, args['visualization_weight'])

train_opt = read_train_options(visualization_weight, train_opt_log_dir)
mlp = train_opt['mlp']
cnn = train_opt['cnn']
image_set = train_opt['image_set']
image_dir = get_image_dir(static.image_dirs, image_set)
resize_size = int(train_opt['resize_size'])

gpu_ids = str2int(train_opt['gpu_ids'])
device = set_device(gpu_ids)


# Target images of Grad-CAM
visualization_split_list = args['visualization_split'].split(',')
df_split = get_column_value(df_source, 'split', visualization_split_list)


# Configure of model
model = CreateModel_MLPCNN(mlp, cnn, num_inputs, num_classes, gpu_ids)
weight = torch.load(visualization_weight)
model.load_state_dict(weight)

# Extract the substance of model when using DataParallel
# eg.
# in case of model_name='Resnet18/50'
# target_layer = model.layer4        when use cpu.
# target_layer = model.module.layer4 when use Dataparallel.
# Target layer of each model
raw_model = model.module if len(gpu_ids) > 0 else model

# Extract CNN from MLP+CNN
if not(cnn is None):
    model_name = cnn
    raw_model = raw_model.cnn
else:
    print('Not CNN')
    exit()


if model_name.startswith('B'):  # EfficientNet
    configs = [dict(model_type='efficientnet', arch=raw_model, layer_name='_conv_head')]

elif model_name.startswith('DenseNet'):
    configs = [dict(model_type='densenet', arch=raw_model, layer_name='features_denseblock4_denselayer24')]

elif model_name.startswith('ResNet'):
    configs = [dict(model_type='resnet', arch=raw_model, layer_name='layer4')]


basename = get_basename(visualization_weight)
target_dir = os.path.join(visualization_dir, basename)
os.makedirs(target_dir, exist_ok=True)


#def visualize():
i = 0
total = len(df_split)
for img_file in df_split['path_to_img']:
    img_path = os.path.join(image_dir, img_file)

    pil_img = PIL.Image.open(img_path).convert('RGB')

    torch_img = transforms.Compose([ transforms.Resize((resize_size, resize_size)),
                                     transforms.ToTensor()
                                    ])(pil_img).to(device)

    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

    for config in configs:
        config['arch'].to(device).eval()

    cams = [
              [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
              for config in configs
            ]

    images = []
    for gradcam, gradcam_pp in cams:
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

        grid_image = make_grid(images, nrow=5)


    # Save saliency map
    save_img_filename = img_file.replace('/', '_').replace(' ', '_')
    save_path = os.path.join(target_dir, save_img_filename)
    transforms.ToPILImage()(grid_image).save(save_path)

    print('{index}/{total}: Exporting saliency map for {img_file}'.format(index=i, total=total, img_file=img_file))
    i = i + 1


# ----- EOF -----
