# Nervus
Nervus can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.


# Preparing
## CSV
<<<<<<< HEAD
CSV must contain columns named 'id_XXX', 'filepath', 'output_XXX', and 'split'.
=======
CSV must contain columns named 'id_XXX', 'filename', 'filepath', 'output_XXX', and 'split'.
>>>>>>> a968d889195bbec3e74d199485154d3ec6fea19d

Note 'id_XXX' must be unique.

When you use inputs other than image, 'input_XXX' is needed. 
When you use deepsurv, 'periords_XXX' is needed as well.

## Model development
For training, validation, and testing, `hyperparameter.csv` and `work_all.sh` should be modified.

GPU and path to `hyperparameter.csv` should be defined in the `work_all.sh`.
Other parameters are defined in the `hyperparameter.csv`. 


# Task
## Single-label/Multi-label output classification, regression, or deepsurv.
Use `train.py`, `test.py` for all task. And also, `evaluation/roc.py`, `evaluation/yy.py` or `evaluation/c_index.py` are used depending on task.


# Debugging
## MakeFile
Edit Makefile according to task.


# CUDA VERSION
CUDA Version = 11.3, 11.4
