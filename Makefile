# ----- Define variables -----
# CSV_NAME = trial.csv
# IMAGE_DIR = 8bit/128 | 8bit/covid | 8bit/png256
# TASK = classification | regression | deepsurv
# MODEL = MLP | [CNN or ViT name] | MLP+[CNN or ViT name]
# CRITERION = CEL | MSE | RMSE | MAE | NLL
# OPTIMIZER = SGD | Adadelta | RMSprop | Adam | RAdam
# AUGMENTATION = xrayaug | trivialaugwide | randaug | no
# SAMPLER = yes | no   # should be no when regression or multi-label
# IN_CHANNEL = 1 | 3
# SAVE_WEIGHT = best | each
# GPU_IDS = -1 | 0-1-2

CSV_NAME := trial.csv
IMAGE_DIR := 128
TASK := classification
MODEL := ResNet18
CRITERION := CEL
OPTIMIZER := Adam
EPOCHS := 3
BATCH_SIZE := 64
SAMPLER := no
AUGMENTATION := xrayaug
IN_CHANNEL := 3
SAVE_WEIGHT := each
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
--in_channel $(IN_CHANNEL) \
--save_weight $(SAVE_WEIGHT) \
--gpu_ids $(GPU_IDS)


PYTHON := python -i
#PYTHON := python

TRAIN_CODE := train.py
TEST_CODE := test.py
EVAL_CODE := eval.py

PARAMETER := ./parameters.csv
RESULTS_DIR := ./results
SETS_DIR := $(RESULTS_DIR)/sets
SUMMARY_DIR := $(RESULTS_DIR)/summary
LOG_DIR := ./logs
TMP_DIR := tmp

DATETIME := $$(date "+%Y-%m-%d-%H-%M-%S")
LOG_DATETIME := $$(ls -d $(SETS_DIR)/[0-9]*)


active:
	pipenv shell

run:
	pipenv run sh work_all.sh

train:
	$(PYTHON) $(TRAIN_CODE) $(TRAIN_OPT)

test:
	$(PYTHON) $(TEST_CODE)

eval:
	$(PYTHON) $(EVAL_CODE)

temp:
	-mkdir -p $(SETS_DIR)/$(TMP_DIR)
	-mkdir -p $(SUMMARY_DIR)/$(TMP_DIR)
	-mkdir -p $(LOG_DIR)/$(TMP_DIR)

clean:
	-mv $(LOG_DATETIME) $(SETS_DIR)/$(TMP_DIR)
	-mv $(SUMMARY_DIR)/summary.csv $(SUMMARY_DIR)/$(TMP_DIR)/summary_$(DATETIME).csv
	-mv $(LOG_DIR)/*.log $(LOG_DIR)/$(TMP_DIR)
