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

from visualization.gradcam.utils import visualize_cam
from visualization.gradcam.gradcam import GradCAM, GradCAMpp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from lib.align_env import *
from config.model import *
from options.visualize_option import VisualizeOptions


args = VisualizeOptions().parse()


dirs_dict = set_dirs()
train_opt_log_dir = dirs_dict['train_opt_log']
weight_dir = dirs_dict['weight']
visualization_dir = dirs_dict['visualization']

visualization_split_list = (args['visualization_split']).split(',')

# Retrieve training options
path_train_opt = get_target(dirs_dict['train_opt_log'], args['visualization_datetime'])  # the latest train_opt if datatime is None
dt_name = get_dt_name(path_train_opt)
train_opt = read_train_options(path_train_opt)  # Revert csv_name
task = train_opt['task']
mlp = train_opt['mlp']
cnn = train_opt['cnn']
gpu_ids = str2int(train_opt['gpu_ids'])
device = set_device(gpu_ids)

image_dir = os.path.join(dirs_dict['images_dir'], train_opt['image_dir'])

csv_dict = parse_csv(os.path.join(dirs_dict['csvs_dir'], train_opt['csv_name']), task)
label_list = csv_dict['label_list']
num_inputs = csv_dict['num_inputs']
num_classes = csv_dict['num_classes']
df_split = get_column_value(csv_dict['source'], 'split', visualization_split_list)


visualization_weight = get_target(weight_dir, dt_name)


# Configure of model
model = CreateModel_MLPCNN(mlp, cnn, label_list, num_inputs, num_classes, device=gpu_ids)
weight = torch.load(visualization_weight)
model.load_state_dict(weight)

# Extract the substance of model when using DataParallel
# eg.
# in case of model_name='Resnet18/50'
# target_layer = model.layer4        when use cpu.
# target_layer = model.module.layer4 when use Dataparallel.
# Target layer of each model
raw_model = model.module if len(gpu_ids) > 0 else model

# Extract CNN from MLP+CNN or CNN
if (mlp is None) and not(cnn is None):
    # CNN only
    model_name = cnn
    if len(label_list) > 1:
        # When multi
        raw_model = raw_model.extractor
    else:
        raw_model = raw_model

elif not(mlp is None) and not(cnn is None):
# Extract CNN from MLP+CNN
    model_name = cnn
    raw_model = raw_model.cnn
else:
    print('\nNot CNN\n')
    exit()

if model_name.startswith('B'):
    # EfficientNet
    configs = [dict(model_type='efficientnet', arch=raw_model, layer_name='8')]

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
for img_file in df_split['dir_to_img']:
    img_path = os.path.join(image_dir, img_file)

    pil_img = PIL.Image.open(img_path).convert('RGB')

    # No rezise
    _transforms = transforms.Compose(
                                     [ transforms.ToTensor() ]
                                    )
    
    torch_img = _transforms(pil_img).to(device)
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
