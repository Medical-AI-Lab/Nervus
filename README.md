# Submitting articles
This model is used in [XXX] published in [YYY] (These will be replaced once publication is finalized).  
In this article, a regression model with ConvNext was applied.  
The best performing model was obtained with a loss function of RMSE.  
Detailed model descriptions are provided in the article and the actual usage of the framework is provided below.

# Nervus
This is an AI model framework used for single/multi-label and/or single/multi-class tasks with image and/or tabular data.
Although this has a possibility to apply wide range of fields, we intended to use this model for medical imaging classification task.

Additionally, we merged DeepSurv model [https://doi.org/10.1186/s12874-018-0482-1] into this model, which is a model that merges Cox proportional hazard model with deep learning. 
It is a useful model for prognosis estimation by dealing with binary variables such as decease or not, and the period until the event. 
The original DeepSurv model could only handle tabular data, but we have added images to it.

Nervus can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.

# Dataset preparation
## Directory tree
Set dataset directories as follows.
  
datasets (Any names are available)  
　　└imgs (Any names are available. This repository has image files for CNN.)  
　　└docs (Any names are available. This repository contains a csv)  
　　　└trial.csv (Any names are available. This is the key csv for Nervus)  

## Key csv
This is the csv which we show as trial.csv in the brief usage section.
CSV must contain columns named `uniqID`, `label_XXX`, and `split`. Additionally, if you use images as inputs, you need `imgpath`.

Example of csv in the docs:
| uniqID |             imgpath            | label_cancer | split |
| -----  | ------------------------------ |  ---------   | ----- |
| 0001   | materials/imgs/png_128/AAA.png | malignant    | train |
| 0002   | materials/imgs/png_128/BBB.png | benign       | val   |
| 0003   | materials/imgs/png_128/CCC.png | malignant    | train |
| 0004   | materials/imgs/png_128/DDD.png | malignant    | test  |
| 0005   | materials/imgs/png_128/EEE.png | benign       | train |
| 0006   | materials/imgs/png_128/FFF.png | malignant    | train |
| 0007   | materials/imgs/png_128/GGG.png | benign       | train |
| 0008   | materials/imgs/png_128/HHH.png | benign       | val   |
| 0009   | materials/imgs/png_128/III.png | malignant    | test  |
| :      | :                              | :            | :     |

Note:
- `uniqID` must be unique.
- `imgpath` should have a path to images for the model if you use image as as inputs.
- `label_XXX` should have a classification target. Any name is available. If you use more than two `label_XXX`, it will be automatically recognize multi-label classification and automatically prepare a proper number of classifiers (FCs).
- `split` should have `train`, `val`, and `test`.
- When you use inputs other than image, `input_XXX` is needed.
- When you use deepsurv, `periods_XXX` is needed as well.


## Model development
For training and internal validation(tuning),

`python train.py --task classification --csvpath datasets/docs/trial.csv --model ResNet18 --criterion CEL --optimizer Adam --epochs 50 --batch_size 32 --sampler no --augmentation randaug --pretrained True --in_channel 1 --save_weight_policy best --gpu_ids 0-1-2-3`

### Arguments
- task: task name
  - example: classification, regression, deepsurv
- csvpath: csv filepath name contains labeled training data, validation data, and test data
- model: model name
  - example
    - MLP only: MLP
    - CNN only: ResNet, ResNet18, DenseNet,
    EfficientNetB0, EfficientNetB2, EfficientNetB4, EfficientNetB6, EfficientNetV2s, EfficientNetV2m, EfficientNetV2l, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ViTb16, ViTb32, ViTl16, ViTl32, ViTH14.
    - MLP+CNN : MLP+ResNet, MLP+EfficientNetB0, MLP+ResNet, MLP+ConvNeXtTiny, MLP+ViTb16 ... (combine above)
- criterion: Loss function
  - example:
    - classification: CEL ※CEL=CrossEntropyLoss
    - regression: MSE, RMSE, MAE
    - deepsurv: NLL
- optimizer: optimization algorithm
  - example: SGD, Adadelta, Adam, RMSprop
- epochs: number of training with entire dataset
- bach_size: number of training data in each batch
- sampler: samples elements randomly or not.
  - example: yes, no
  Note that this only works for two-class classification task for now.
- augmentation: increase the amount of data by slightly modified copies or created synthetic.
  - example: trivialaugwide, randaug, and no.
- pretrained: specify True if pretrained model of CNN or ViT is used, otherwise False.
- in_channel: specify the channel of when image is handled, or any of 1 channel(grayscale) and 3 channel(RGB).
  - example:
    - 1 channel(grayscale): 1
    - 3 channel(RGB): 3
- save_weight_policy: specify when you save weights.
  - example:
    - Save the lowest validation loss: best
    - Save each time the loss value is updated: each
- gpu_ids
  - example:
    - No gpu (cpu only): cpu
    - 1 gpu: 0
    - 2 gpus: 0-1
    - 4 gpus: 0-1-2-3


## Model test
For test trained model,

`python test.py --csvpath datasets/docs/trial.csv --weight_dir results/trial/trials/YYYY-MM-DD-HH-mm-ss/weights`

### Arguments
- csvpath: csv filepath name contains test data.
- weight_dir: path to a directory which contains weights

## For many trials
If you need many trials, use `work_all.sh`. In this case, `parameter.csv` must be prepared. Examples are shown in this repository.


# Tutorial
Tutorial for Nervus library is available on Google Colaboratory.
To do the tutorial, please visit this site [https://colab.research.google.com/drive/1710VAktDPVyPZdRo39UrSAtuVBYdFsCT].


# CUDA VERSION
CUDA Version = 11


# Citation
If you find this project useful, please cite our paper:

```
@article{Nervus2022,
  title = {Nervus: A comprehensive deep learning classification, regression, and prognostication tool for both medical image and clinical data analysis},
  author = {Toshimasa Matsumoto, Shannon L Walston, Yukio Miki, Daiju Ueda},
  year = {2022},
  archivePrefix = {arXiv},
  eprint = {2212.11113}
}
```


# Acknowledgement
This project is working in progress, thus the codebase and model might not be perfect or bug-free. We will very much appreciate any kind of contribution or and issue raised. If you find a bug or have any suggestion, please feel free to open an issue or contact us.
