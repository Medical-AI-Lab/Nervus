# Nervus
can handle the following task:
- Single/Multi-label-output classification with any of MLP, CNN, or MLP+CNN.
- Single/Multi-label-output regression with any of MLP, CNN, or MLP+CNN.
- DeepSurv with any of MLP, CNN, or MLP+CNN.

# Preparing
## CSV
CSV must contain columns named 'id_XXX', 'filename', 'filepath', 'label_XXX', and 'split'.

Note 'id_XXX' must be unique.

When you use inputs other than image, 'input_XXX' is needed. 
When you use deepsurv, 'periords_XXX' is needed as well.
## Model development
For training, validation, and testing, `hyperparameter.csv` and `work_all.sh` should be modified.
When task is deepsurv, `hyperparameter_deepsurv.csv` and `work_all_deepsurv.sh` should be modified.

GPU and path to `hyperparameter.csv` should be defined in the `work_all.sh`.
Other parameters are defined in the `hyperparameter.csv`. 

For deepsurv, Similarty to the above.

# Task
## Single-label/Multi-label output classification or regression
Use `train.py`, `test.py`, and `evaluation/roc.py` or `evaluation/yy.py`.

## Deepsurv
In deepsurv(the number of labels = 1 and the number of periods = 1), use `train_deepsurv.py`, `test_depsurv.py`, and `c_index.py`.


# Debugging
## MakeFile
Edit Makefile or Makefile_deepsurv according to task.


# CUDA VERSION
CUDA Version = 11.3, 11.4
