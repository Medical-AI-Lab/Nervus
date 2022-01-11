# ----- Define variables -----
# CSV_NAME = clean.csv(single-label classifiation or deepsurv) | clean_reg.csv | clean_cla_multi.csv | clean_reg_multi.csv
# IMAGE_DIR = 128 | covid | png256
# TASK = classification | regression | deepsurv
# MODEL = MLP | ResNet18 | MLP+ResNet18
# CRITERION = CrossEntropyLoss | MSE | NLL
# SAMPLER = yes | no    # should be no when regression or multi-label
#GPU_IDS = -1 | 0,1,2
CSV_NAME := clean.csv
IMAGE_DIR := 128
TASK := deepsurv
MODEL := MLP
CRITERION := NLL
OPTIMIZER := Adam
EPOCHS := 3
BATCH_SIZE := 64
SAMPLER := no
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
C_INDEX_CODE := ./evaluation/c_index.py
GRADCAM_CODE := visualize.py

# Directory
TRAIN_OPT_LOG_DIR := ./train_opt_logs
HYPERPARAMETER_DIR := ./hyperparameters
WEIGHT_DIR := ./weights
LOG_DIR := ./logs
RESULT_DIR := ./results
LEARNING_CURVE_DIR := $(RESULT_DIR)/learning_curve
LIKELIHOOD_DIR := $(RESULT_DIR)/likelihood
ROC_DIR := $(RESULT_DIR)/roc
ROC_SUMMARY_DIR := $(ROC_DIR)/summary
YY_DIR := $(RESULT_DIR)/yy
YY_SUMMARY_DIR := $(YY_DIR)/summary
C_INDEX_DIR := $(RESULT_DIR)/c_index
C_INDEX_SUMMARY_DIR := $(C_INDEX_DIR)/summary
VISUALIZATION_DIR := $(RESULT_DIR)/visualization
TMP_DIR := tmp
DATETIME := $$(date "+%Y-%m-%d-%H-%M-%S")

temp:
	-mkdir -p $(TRAIN_OPT_LOG_DIR)/$(TMP_DIR)
	@#-mkdir -p $(HYPERPARAMETER_DIR)/$(TMP_DIR)
	-mkdir -p $(TRAIN_OPT_LOG_DIR)/$(TMP_DIR)
	-mkdir -p $(WEIGHT_DIR)/$(TMP_DIR)
	-mkdir -p $(LOG_DIR)/$(TMP_DIR)
	-mkdir -p $(LEARNING_CURVE_DIR)/$(TMP_DIR)
	-mkdir -p $(LIKELIHOOD_DIR)/$(TMP_DIR)
	-mkdir -p $(ROC_DIR)/$(TMP_DIR)
	-mkdir -p $(ROC_SUMMARY_DIR)/$(TMP_DIR)
	-mkdir -p $(YY_DIR)/$(TMP_DIR)
	-mkdir -p $(YY_SUMMARY_DIR)/$(TMP_DIR)
	@#-mkdir -p $(C_INDEX_DIR)/$(TMP_DIR)
	-mkdir -p $(C_INDEX_SUMMARY_DIR)/$(TMP_DIR)
	-mkdir -p $(VISUALIZATION_DIR)/$(TMP_DIR)

clean:
	-mv $(TRAIN_OPT_LOG_DIR)/*.csv $(TRAIN_OPT_LOG_DIR)/$(TMP_DIR)
	-mv $(WEIGHT_DIR)/*.pt $(WEIGHT_DIR)/$(TMP_DIR)
	@#-mv $(HYPERPARAMETER_DIR)/*.csv $(HYPERPARAMETER_DIR)/$(TMP_DIR)
	-mv $(LOG_DIR)/*.log $(LOG_DIR)/$(TMP_DIR)
	-mv $(LEARNING_CURVE_DIR)/*.csv $(LEARNING_CURVE_DIR)/$(TMP_DIR)
	-mv $(LIKELIHOOD_DIR)/*.csv $(LIKELIHOOD_DIR)/$(TMP_DIR)
	-mv $(ROC_DIR)/*.png $(ROC_DIR)/$(TMP_DIR)
	-mv $(ROC_SUMMARY_DIR)/summary.csv $(ROC_SUMMARY_DIR)/$(TMP_DIR)/summary_$(DATETIME).csv
	-mv $(YY_DIR)/*.png $(YY_DIR)/$(TMP_DIR)
	-mv $(YY_SUMMARY_DIR)/summary.csv $(YY_SUMMARY_DIR)/$(TMP_DIR)/summary_$(DATETIME).csv
	@#-mv $(C_INDEX_DIR)/*.png $(C_INDEX_DIR)/$(TMP_DIR)
	-mv $(C_INDEX_SUMMARY_DIR)/summary.csv $(C_INDEX_SUMMARY_DIR)/$(TMP_DIR)/summary_$(DATETIME).csv
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

c_index:
	$(PYTHON) $(C_INDEX_CODE)

gradcam:
	$(PYTHON) $(GRADCAM_CODE)
