#!/usr/bin/env bash


hyperparameter_csv="./hyperparameters/hyperparameter.csv"


train_log="./logs/train.log"
test_log="./logs/test.log"
roc_log="./logs/roc.log"
yy_log="./logs/yy.log"


#python="python3"
python="python"

train_code="train_mlp_cnn.py"
test_code="test_mlp_cnn.py"
roc_code="./evaluation/roc.py"
#yy_code="./evaluation/yy.py"
gpu_ids="-1"

# Delete previous logs.
rm -f ${train_log}
rm -f ${test_log}
rm -f ${roc_log}
rm -f ${yy_log}


total=$(tail -n +2 "${hyperparameter_csv}" | wc -l)
i=1
for row in `tail -n +2 ${hyperparameter_csv}`; do
  model=$(echo "${row}" | cut -d "," -f1)
  criterion=$(echo "${row}" | cut -d "," -f2)
  optimizer=$(echo "${row}" | cut -d "," -f3)
  batch_size=$(echo "${row}" | cut -d "," -f4)
  epochs=$(echo "${row}" | cut -d "," -f5)
  image_set=$(echo "${row}" | cut -d "," -f6)
  resize_size=$(echo "${row}" | cut -d "," -f7)
  sampler=$(echo "${row}" | cut -d "," -f8)
  normalize_image=$(echo "${row}" | cut -d "," -f9)

  echo "${i}/${total}: Training starts..."

  echo ""

  # Traning
  echo "${python} ${train_code} --model ${model} --criterion ${criterion} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --image_set ${image_set} --resize_size ${resize_size} --sampler ${sampler} --normalize_image ${normalize_image} --gpu_ids ${gpu_ids}"
  ${python} ${train_code} --model ${model} --criterion ${criterion} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --image_set ${image_set} --resize_size ${resize_size} --sampler ${sampler} --normalize_image ${normalize_image} --gpu_ids ${gpu_ids} 2>&1 | tee -a ${train_log}

  echo ""

  # Test
  echo "${i}/${total}: Test starts..."
  echo "${python} ${test_code}"
  ${python} ${test_code} 2>&1 | tee -a ${test_log}

  echo ""

  # Classification
  # Plot ROC
  echo "${i}/${total}: Plot ROC..."
  echo "${python} ${roc_code}"
  ${python} ${roc_code} 2>&1 | tee -a ${roc_log}


  # Regression
  # Plot yy-graph
  #echo "${i}/${total}: Plot yy-graph..."
  #echo "python ${yy_code}"
  #python ${yy_code} |& tee -a ${yy_log_file}


  i=$(($i + 1))
  echo -e "\n"

done
