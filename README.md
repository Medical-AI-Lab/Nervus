# Nervus
Nervus can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.

# Overview
This is an AI model used for single/multi-label and/or single/multi-class tasks with image and/or tabular data.
Although this has a possibility to apply wide range of fields, we inteded to use this model for medical imaging classification task.

Additionally, we merged DeepSurv model [citation] into this model, which is a model that merges Cox proportional hazard model with deep learning. It is a useful model for prognosis estimation by dealing with binary variables such as deseace or not, and the period until the event. The original DeepSurv model could only handle tabular data, but we have added images to it.

# Beief Usage
- Directory tree  
Set directories as follows.  

┌Nervus (this repository)  
└materials   
　└images (this repository has image files for CNN.)  
　└csvs  
　　　 └trials.csv (any name is available if you change `hyperparameters/hyperparameters.csv`)

- Brief modification for your taks
  - hyperparameters/hyperparameters.csv
    CSV must contain columns named `id_XXX`, `filepath`, `output_XXX`, and `split`.  
    Detailed explanation is shown in below.
  - work_all.sh
    Change `gpu_ids` depending on how many GPUs you can use. Default is "-1" which means to use CPU only.

- To work and evaluate  
`$bash work_all.sh`

# Detailed Preparation
## CSV
CSV must contain columns named `id_XXX`, `filepath`, `output_XXX`, and `split`.

Examples:
id_uniq, filepath, output_cancer, split
0001, png/AAA.png, malignant, train
0002, png/BBB.png, bening, val
0003, png/CCC.png, bening, test
0004, png/DDD.png, malignant, train
:
:
:

Note `id_XXX` must be unique.
`filepath` should have a path to images for the model.
`output_XXX` should have a classification target. Any name is available. If you use more than two `output_XXX`, it will be automatically recognize multi-label classification and automatically prepare a proper number of classifiers (FCs). 
When you use inputs other than image, `input_XXX` is needed. 
When you use deepsurv, `periords_XXX` is needed as well.

## Model development
For training, validation, and testing, `hyperparameter.csv` and `work_all.sh` should be modified.

GPU and path to `hyperparameter.csv` should be defined in the `work_all.sh`.
Other parameters are defined in the `hyperparameter.csv`. 

### hyperparameter.csv items
- task: task name
  - example: classification, regression, deepsurv
- csv_name: csv file name contains labeled training data, validation data, and test data
  - default path: ../materials/csvs/ (see: align_env.py)
- image_dir: image dataset directory name
  - default path: ../materials/images/ (see: align_env.py)
- model: model name
  - example
    - MLP only: MLP
    - CNN only: B0, B2, B4, B6, ResNet, ResNet18, DenseNet
    - MLP+CNN : MLP+B0, MLP+ResNet, ... (combine above)
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

# Task
## Single-label/Multi-label output classification, regression, or deepsurv.
For all task, `train.py` and `test.py` are used. And also, `evaluation/roc.py`, `evaluation/yy.py` or `evaluation/c_index.py` are used depending on task.


# Debugging
## MakeFile
Edit Makefile according to task.

## env
'''
~/lib/align_env.py
'''

# CUDA VERSION
CUDA Version = 11.3, 11.4
