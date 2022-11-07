# ----- Define variables -----
# TASK = classification | regression | deepsurv
# CSVPATH = materials/docs/trial.csv
# MODEL = MLP | [CNN or ViT name] | MLP+[CNN or ViT name]
# CRITERION = CEL | MSE | RMSE | MAE | NLL
# OPTIMIZER = SGD | Adadelta | RMSprop | Adam | RAdam
# EPOCHS  = 50
# BATCH_SIZE = 64
# SAMPLER = yes | no   # should be no when regression or multi-label
# AUGMENTATION = xrayaug | trivialaugwide | randaug | no
# IN_CHANNEL = 1 | 3
# SAVE_WEIGHT_POLICY = best | each
# GPU_IDS = cpu | 0-1-2

TASK := classification
CSVPATH := materials/docs/trial.csv
MODEL := MLP+ResNet18
CRITERION := CEL
OPTIMIZER := Adam
EPOCHS := 50
BATCH_SIZE := 64
SAMPLER := no
AUGMENTATION := xrayaug
IN_CHANNEL := 1
SAVE_WEIGHT_POLICY := each
GPU_IDS := cpu

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
--in_channel $(IN_CHANNEL) \
--save_weight_policy $(SAVE_WEIGHT_POLICY) \
--gpu_ids $(GPU_IDS)

TEST_OPT := \
--csvpath $(CSVPATH)

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
