# ----- Define variables -----
# TASK = classification | regression | deepsurv
# CSVPATH = materials/docs/trial.csv
# MODEL = MLP | [CNN or ViT] | MLP+[CNN or ViT]
# CRITERION = CEL | MSE | RMSE | MAE | NLL
# OPTIMIZER = SGD | Adadelta | RMSprop | Adam | RAdam
# EPOCHS = 50
# BATCH_SIZE = 64
# SAMPLER = weighted | distributed | distweight | no
# AUGMENTATION = xrayaug | trivialaugwide | randaug | no
# BIT_DEPTH = 8 | 16
# IN_CHANNEL = 1 | 3
# PRETRAINED = True | False
# SAVE_WEIGHT_POLICY = best | each
# TRAIN_GPU_IDS = cpu | 0-1-2
# TEST_BATCH_SIZE = 64
# TEST_GPU_IDS = cpu | 0-1-2

TASK := classification
CSVPATH := materials/dogcat/docs/tutorial_src_dog.csv
MODEL := ResNet18
CRITERION := CEL
OPTIMIZER := Adam
EPOCHS := 50
BATCH_SIZE := 64
SAMPLER := no
AUGMENTATION := no
BIT_DEPTH := 8
IN_CHANNEL := 1
PRETRAINED := False
SAVE_WEIGHT_POLICY := each
TRAIN_GPU_IDS := cpu

TEST_BATCH_SIZE := $(BATCH_SIZE)
TEST_GPU_IDS := $(TRAIN_GPU_IDS)


TRAIN_OPT := \
--task $(TASK) \
--csvpath $(CSVPATH) \
--model $(MODEL) \
--criterion $(CRITERION) \
--optimizer $(OPTIMIZER) \
--epochs $(EPOCHS) \
--batch_size $(BATCH_SIZE) \
--sampler $(SAMPLER) \
--augmentation $(AUGMENTATION) \
--pretrained $(PRETRAINED) \
--bit_depth $(BIT_DEPTH) \
--in_channel $(IN_CHANNEL) \
--save_weight_policy $(SAVE_WEIGHT_POLICY) \
--gpu_ids $(TRAIN_GPU_IDS)

TEST_OPT := \
--csvpath $(CSVPATH) \
--test_batch_size $(TEST_BATCH_SIZE) \
--gpu_ids $(TEST_GPU_IDS)


PYTHON := python
TRAIN_CODE := train.py
TEST_CODE := test.py
EVAL_CODE := eval.py


active:
	pipenv shell

run:
	pipenv run sh work_all.sh

train:
	$(PYTHON) $(TRAIN_CODE) $(TRAIN_OPT)

test:
	$(PYTHON) $(TEST_CODE) $(TEST_OPT)

eval:
	$(PYTHON) $(EVAL_CODE)
