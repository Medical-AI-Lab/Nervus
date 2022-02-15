# Nervus
Nervus can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.


# Preparing
## CSV
CSV must contain columns named 'id_XXX', 'filepath', 'output_XXX', and 'split'.

Note 'id_XXX' must be unique.

When you use inputs other than image, 'input_XXX' is needed. 
When you use deepsurv, 'periords_XXX' is needed as well.

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
