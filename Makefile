# ----- Define variables -----
# CSV_NAME = clean.csv
# IMAGE_DIR = 128 | covid | png256
# TASK = classification | regression
# MODEL = MLP | ResNet18 | MLP+ResNet18
# CRITERION = CrossEntropyLoss | MSE
# SAMPLER = yes | no    # should be no when regression or multi-label
#GPU_IDS = -1 | 0,1,2
CSV_NAME := clean.csv     
IMAGE_DIR := 128
TASK := classification
MODEL := MLP
CRITERION := CrossEntropyLoss
OPTIMIZER := Adam
EPOCHS := 3
BATCH_SIZE := 64
SAMPLER := yes
GPU_IDS := -1

TRAIN_OPT := \
--csv_name $(CSV_NAME) \
--image_dir $(IMAGE_DIR) \
--task $(TASK) \
--model $(MODEL) \
--criterion $(CRITERION) \
--optimizer $(OPTIMIZER) \
--epochs $(EPOCHS) \
--batch_size $(BATCH_SIZE) \
--sampler $(SAMPLER) \
--gpu_ids $(GPU_IDS)



PYTHON := python -i
#PYTHON := python

TRAIN_CODE := train.py
TEST_CODE := test.py
ROC_CODE := ./evaluation/roc.py
YY_CODE := ./evaluation/yy.py
GRADCAM_CODE := visualize.py


# Directory
TRAIN_OPT_LOG_DIR := ./train_opt_logs
WEIGHT_DIR := ./weights
LOG_DIR := ./logs

RESULT_DIR := ./results
LEARNING_CURVE_DIR := $(RESULT_DIR)/learning_curve
LIKELIHOOD_DIR := $(RESULT_DIR)/likelihood
ROC_DIR := $(RESULT_DIR)/roc
YY_DIR := $(RESULT_DIR)/yy
VISUALIZATION_DIR := $(RESULT_DIR)/visualization
TMP_DIR := tmp




temp:
	-mkdir -p $(TRAIN_OPT_LOG_DIR)/$(TMP_DIR)
	-mkdir -p $(HYPERPARAMETER_DIR)/$(TMP_DIR)
	-mkdir -p $(TRAIN_OPT_LOG_DIR)/$(TMP_DIR)
	-mkdir -p $(WEIGHT_DIR)/$(TMP_DIR)
	-mkdir -p $(LOG_DIR)/$(TMP_DIR)
	-mkdir -p $(LEARNING_CURVE_DIR)/$(TMP_DIR)
	-mkdir -p $(LIKELIHOOD_DIR)/$(TMP_DIR)
	-mkdir -p $(ROC_DIR)/$(TMP_DIR)
	-mkdir -p $(YY_DIR)/$(TMP_DIR)
	-mkdir -p $(VISUALIZATION_DIR)/$(TMP_DIR)


clean:
	-mv $(TRAIN_OPT_LOG_DIR)/*.csv $(TRAIN_OPT_LOG_DIR)/$(TMP_DIR)
	-mv $(WEIGHT_DIR)/*.pt $(WEIGHT_DIR)/$(TMP_DIR)
	@#-mv $(HYPERPARAMETER_DIR)/*.csv $(HYPERPARAMETER_DIR)/$(TMP_DIR)
	-mv $(LOG_DIR)/*.log $(LOG_DIR)/$(TMP_DIR)
	-mv $(LEARNING_CURVE_DIR)/*.csv $(LEARNING_CURVE_DIR)/$(TMP_DIR)
	-mv $(LIKELIHOOD_DIR)/*.csv $(LIKELIHOOD_DIR)/$(TMP_DIR)
	-mv $(ROC_DIR)/*.png $(ROC_DIR)/$(TMP_DIR)
	-mv $(YY_DIR)/*.png $(YY_DIR)/$(TMP_DIR)
	-mv $(VISUALIZATION_DIR)/*_* $(VISUALIZATION_DIR)/$(TMP_DIR)


active:
	pipenv shell


train:
	$(PYTHON) $(TRAIN_CODE) $(TRAIN_OPT)


test:
	$(PYTHON) $(TEST_CODE)


roc:
	$(PYTHON) $(ROC_CODE)


yy:
	$(PYTHON) $(YY_CODE)


gradcam:
	$(PYTHON) $(GRADCAM_CODE)
