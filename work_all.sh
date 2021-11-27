#!/usr/bin/env bash


hyperparameter_csv="./hyperparameters/hyperparameter.csv"


train_log="./logs/train.log"
test_log="./logs/test.log"
roc_log="./logs/roc.log"
yy_log="./logs/yy.log"


#python="python3"
python="python"

train_code="train.py"
test_code="test.py"
roc_code="./evaluation/roc.py"
yy_code="./evaluation/yy.py"
gpu_ids="-1"

# Delete previous logs.
rm -f "${train_log}"
rm -f "${test_log}"
rm -f "${roc_log}"
rm -f "${yy_log}"

#1 task,
#2 csv_name,
#3 image_dir,
#4 model,
#5 criterion,
#6 optimizer,
#7 epochs,
#8 batch_size,
#9 sampler

total=$(tail -n +2 "${hyperparameter_csv}" | wc -l)
i=1
for row in $(tail -n +2 "${hyperparameter_csv}"); do
  task=$(echo "${row}" | cut -d "," -f1)
  csv_name=$(echo "${row}" | cut -d "," -f2)
  image_dir=$(echo "${row}" | cut -d "," -f3)
  model=$(echo "${row}" | cut -d "," -f4)
  criterion=$(echo "${row}" | cut -d "," -f5)
  optimizer=$(echo "${row}" | cut -d "," -f6)
  epochs=$(echo "${row}" | cut -d "," -f7)
  batch_size=$(echo "${row}" | cut -d "," -f8)
  sampler=$(echo "${row}" | cut -d "," -f9)


  echo "${i}/${total}: Training starts..."

  echo ""

  # Traning
  echo "${python} ${train_code} --task ${task} --csv_name ${csv_name} --image_dir ${image_dir} --model ${model} --criterion ${criterion} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --sampler ${sampler} --gpu_ids ${gpu_ids}"
  "${python}" "${train_code}" --task "${task}" --csv_name "${csv_name}" --image_dir "${image_dir}" --model "${model}" --criterion "${criterion}" --optimizer "${optimizer}" --epochs "${epochs}" --batch_size "${batch_size}" --sampler "${sampler}" --gpu_ids "${gpu_ids}" 2>&1 | tee -a "${train_log}"

  echo ""

  # Test
  echo "${i}/${total}: Test starts..."
  echo "${python} ${test_code}"
  "${python}" "${test_code}" 2>&1 | tee -a "${test_log}"

  echo ""

  # Classification
  # Plot ROC
  echo "${i}/${total}: Plot ROC..."
  echo "${python} ${roc_code}"
  "${python}" "${roc_code}" 2>&1 | tee -a "${roc_log}"


  # Regression
  # Plot yy-graph
  #echo "${i}/${total}: Plot yy-graph..."
  #echo "python ${yy_code}"
  #python ${yy_code} |& tee -a ${yy_log_file}


  i=$(($i + 1))
  echo -e "\n"

done
