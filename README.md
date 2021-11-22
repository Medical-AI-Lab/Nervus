# Nervus
Classification with any of MLP, CNN, or MLP+CNN.

# Preparing
## CSV
CSV must contain columns named 'id_XXX, ', 'filename', 'dir_to_image', 'input_XXX', 'label_XXX', and 'split'.
## Model development
For training, validation, testing, hyperparameter.csv and work_all.sh should be modified.

GPU and path to hyperparameter.csv should be defined in the work_all.sh.
Other parameters are defined in the hyperparameter.csv. 

# Debugging
## MakeFile
Edit Makefile according to your environment and situation.
