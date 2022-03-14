# ----- Define variables -----
# CSV_NAME = trials.csv
# IMAGE_DIR = 128 | covid | png256
# TASK = classification | regression | deepsurv
# MODEL = MLP | ResNet18 | MLP+ResNet18
# CRITERION = CEL | MSE | NLL
# SAMPLER = yes | no   # should be no when regression or multi-label
# GPU_IDS = -1 | 0,1,2
CSV_NAME := trials.csv
IMAGE_DIR := 128
TASK := classification
MODEL := MLP+ResNet18
CRITERION := CEL
OPTIMIZER := Adam
EPOCHS := 3
BATCH_SIZE := 64
SAMPLER := no
AUGMENTATION := yes
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
--augmentation $(AUGMENTATION) \
--gpu_ids $(GPU_IDS)


PYTHON := python -i
#PYTHON := python

TRAIN_CODE := train.py
TEST_CODE := test.py
ROC_CODE := ./evaluation/roc.py
YY_CODE := ./evaluation/yy.py
C_INDEX_CODE := ./evaluation/c_index.py

# Directory
PARAMETER_DIR := ./parameters
RESULTS_DIR := ./results
SETS_DIR := $(RESULTS_DIR)/sets
SUMMARY_DIR := $(RESULTS_DIR)/summary
LOG_DIR := ./logs
TMP_DIR := tmp

DATETIME := $$(date "+%Y-%m-%d-%H-%M-%S")
LOG_DATETIME := $$(ls -d $(SETS_DIR)/[0-9]*)

temp:
	-mkdir -p $(SETS_DIR)/$(TMP_DIR)
	-mkdir -p $(SUMMARY_DIR)/$(TMP_DIR)
	-mkdir -p $(LOG_DIR)/$(TMP_DIR)
#	@#-mkdir -p $(PARAMETER_DIR)/$(TMP_DIR)

clean:
	-mv $(LOG_DATETIME) $(SETS_DIR)/$(TMP_DIR)
	-mv $(SUMMARY_DIR)/summary.csv $(SUMMARY_DIR)/$(TMP_DIR)/summary_$(DATETIME).csv
	-mv $(LOG_DIR)/*.log $(LOG_DIR)/$(TMP_DIR)
#	@#-mv $(PARAMETER_DIR)/*.csv $(PARAMETER_DIR)/$(TMP_DIR)


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
