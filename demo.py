#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd
import torch

from config.model import *
from lib import *
from options import TestOptions

logger = NervusLogger.get_logger('model_demo')
## remove comment out when debug
NervusLogger.set_level(logging.DEBUG)

nervusenv = NervusEnv()
args = TestOptions().parse()
datetime_dir = get_target(nervusenv.sets_dir, args['test_datetime'])   # args['test_datetime'] if exists or the latest
parameters_path = os.path.join(datetime_dir, nervusenv.csv_parameters)
train_parameters = read_train_parameters(parameters_path)
task = train_parameters['task']
mlp = train_parameters['mlp']
cnn = train_parameters['cnn']
gpu_ids = str2int(train_parameters['gpu_ids'])
device = set_device(gpu_ids)

sp = SplitProvider(os.path.join(nervusenv.splits_dir, train_parameters['csv_name']), task)

# Align option for test only
test_weight = os.path.join(datetime_dir, nervusenv.weight)
train_parameters['preprocess'] = 'no'                           # MUST: Stop augmentaion, No need when test
train_parameters['normalize_image'] = args['normalize_image']   # Default: 'yes'

## bool of using neural network
hasMLP = mlp is not None
hasCNN = cnn is not None

# Configure of model
model = create_mlp_cnn(mlp, cnn, sp.num_inputs, sp.num_classes_in_internal_label, gpu_ids=gpu_ids)
weight = torch.load(test_weight)
model.load_state_dict(weight)

import gradio as gr

def predict_image(input_img):
    model.eval()
    with torch.no_grad():
        inputs_values_normed = []
        # image = transforms.ToTensor()(input_img).unsqueeze(0)
        image = convert_image(input_img)
        # image = image.expand(1, 3, 128, 128)
        image = image.unsqueeze(0)
        logger.debug(image.size())
        outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, image)
        # logger.debug(outputs)

    return pd.DataFrame(outputs.detach().numpy())

def convert_image(image=""):
    transform = _make_transform()
    image = transform(image)      # transform, ie. To_Tensor() and Normalization

    return image

import torchvision.transforms as transforms

def _make_transform():
    _transforms = []

    # MUST: Always convert to Tensor
    _transforms.append(transforms.ToTensor())   # PIL -> Tensor

    if args['normalize_image'] == 'yes':
        # Normalize accepts Tensor only
        _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    _transforms = transforms.Compose(_transforms)
    return _transforms

if __name__=="__main__":
    iface = gr.Interface(predict_image,
        gr.inputs.Image(type="pil"),
        "dataframe")
    iface.launch()
