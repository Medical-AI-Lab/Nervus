# Nervus
can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.

# Preparing
## CSV
CSV must contain columns named 'id_XXX, ', 'filename', 'dir_to_image', 'input_XXX', 'label_XXX', and 'split'.
And also, 'periords_XXX' for deepsurv.
## Model development
For training, validation, testing, hyperparameter.csv and work_all.sh should be modified.
When task is deepsurv, hyperparameter_deepsurv.csv and work_all_deepsurv.sh should be modified.

GPU and path to hyperparameter.csv should be defined in the work_all.sh.
Other parameters are defined in the hyperparameter.csv. 

For deepsurv, Similarty to the above.

# Task
## Single-label output classification/regression
In single-label output classification or regression(the number of labels = 1), use train.py, test.py, and evaluation/roc.py or evaluation/yy.py.

## Multi-label output classifiucation/regression
In multi-label output classification or regression(the number of labels >= 2), use train_multi.py, test_multi.py, and roc_multi.py or yy_multi.py.

## Deepsurv
In deepsurv(the number of labels = 1 and the number of periods = 1), use train_deepsurv.py, test_depsurv.py, and c_index.py.


# Debugging
## MakeFile
Edit Makefile or Makefile_deepsurv according to task.


# CUDA VERSION
CUDA Version = 11.3, 11.4


