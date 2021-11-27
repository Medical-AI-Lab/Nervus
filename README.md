# Nervus
Classification with any of MLP, CNN, or MLP+CNN.

# Preparing
## CSV
CSV must contain columns named 'id_XXX, ', 'filename', 'dir_to_image', 'input_XXX', 'label_XXX', and 'split'.
## Model development
For training, validation, testing, hyperparameter.csv and work_all.sh should be modified.

GPU and path to hyperparameter.csv should be defined in the work_all.sh.
Other parameters are defined in the hyperparameter.csv. 


# Single-label output classification/regression or Multi-label output classifiucation/regression
In single-label output classification or regression(the number of labels = 1), use train.py, test.py, roc.py and yy.py.

In multi-label output classification or regression(the number of labels >= 2), use train_multi.py, test_multi.py, roc_multi.py and yy_multi.py,

Other python codes are commonly used in both of single-label output and multi-label output.


# Debugging
## MakeFile
Edit Makefile according to your environment and situation.


# CUDA VERSION
CUDA Version = 11.3, 11.4


