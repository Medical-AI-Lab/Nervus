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
nervusenv = NervusEnv()
datetime_dir = get_target(nervusenv.sets_dir, args['visualization_datetime'])   # args['test_datetime'] if exists or the latest
hyperparameters_path = os.path.join(datetime_dir, nervusenv.csv_hyperparameters)
train_hyperparameters = read_train_hyperparameters(hyperparameters_path)
task = train_hyperparameters['task']
mlp = train_hyperparameters['mlp']
cnn = train_hyperparameters['cnn']
gpu_ids = str2int(train_hyperparameters['gpu_ids'])
device = set_device(gpu_ids)

image_dir = os.path.join(nervusenv.images_dir, train_hyperparameters['image_dir'])
csv_dict = parse_csv(os.path.join(nervusenv.csvs_dir, train_hyperparameters['csv_name']), task)
label_num_classes = csv_dict['label_num_classes']
num_inputs = csv_dict['num_inputs']
label_list = csv_dict['label_list']

visualization_split_list = args['visualization_split'].split(',')
df_split = get_column_value(csv_dict['source'], 'split', visualization_split_list)
df_split_image = df_split[csv_dict['filepath_column']]

# Configure of model
model = create_mlp_cnn(mlp, cnn, num_inputs, label_num_classes, gpu_ids=gpu_ids)
path_visualization_weight = os.path.join(datetime_dir, nervusenv.weight)
weight = torch.load(path_visualization_weight)
model.load_state_dict(weight)

# Extract the substance of model when using DataParallel
# eg.
# in case of model_name='Resnet18/50'
# target_layer = model.layer4        when use cpu.
# target_layer = model.module.layer4 when use Dataparallel.
# Target layer of each model
raw_model = model.module if len(gpu_ids) > 0 else model

# Extract CNN from MLP+CNN or CNN
assert (cnn is not None), 'Not CNN.'
if (mlp is None):
    # CNN only
    model_name = cnn
    if len(label_list) > 1:
        # When multi
        raw_model = raw_model.extractor
    else:
        raw_model = raw_model
else:
    # Extract CNN from MLP+CNN
    model_name = cnn
    raw_model = raw_model.cnn

if model_name.startswith('B'):
    configs = [dict(model_type='efficientnet', arch=raw_model, layer_name='8')]
elif model_name.startswith('DenseNet'):
    configs = [dict(model_type='densenet', arch=raw_model, layer_name='features_denseblock4_denselayer24')]
elif model_name.startswith('ResNet'):
    configs = [dict(model_type='resnet', arch=raw_model, layer_name='layer4')]


visualization_dir = os.path.join(datetime_dir, nervusenv.visualization_dir)
os.makedirs(visualization_dir, exist_ok=True)

i = 0
total = len(df_split_image)
for img_file in df_split_image:
    img_path = os.path.join(image_dir, img_file)
    pil_img = PIL.Image.open(img_path).convert('RGB')

    # No rezise
    _transforms = transforms.Compose([
                                        transforms.ToTensor()
                                    ])
    
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
    save_path = os.path.join(visualization_dir, save_img_filename)
    transforms.ToPILImage()(grid_image).save(save_path)

    print(f"{i}/{total}: Exporting saliency map for {img_file}")
    i = i + 1

# ----- EOF -----
