#!/usr/bin/env bash

set -eu

gpu_ids="-1"
parameter_csv="./parameters/parameter.csv"

train_log="./logs/train.log"
test_log="./logs/test.log"
roc_log="./logs/roc.log"
yy_log="./logs/yy.log"
c_index_log="./logs/c_index.log"


#python="python3"
python="python"

train_code="train.py"
test_code="test.py"
roc_code="./evaluation/roc.py"
yy_code="./evaluation/yy.py"
c_index_code="./evaluation/c_index.py"

# Delete previous logs.
rm -f "$train_log"
rm -f "$test_log"
rm -f "$roc_log"
rm -f "$yy_log"
rm -f "$c_index_log"

while getopts ":g" optKey; do
  case "$optKey" in
    g)
      gpu_ids="0"
      ;;
  esac
done

#1  task,
#2  csv_name,
#3  image_dir,
#4  model,
#5  criterion,
#6  optimizer,
#7  epochs,
#8  batch_size,
#9  sampler,
#10 augmtntation,
#11 input_channel
total=$(tail -n +2 "$parameter_csv" | wc -l)
i=1
for row in $(tail -n +2 "$parameter_csv"); do
  task=$(echo "$row" | cut -d "," -f1)
  csv_name=$(echo "$row" | cut -d "," -f2)
  image_dir=$(echo "$row" | cut -d "," -f3)
  model=$(echo "$row" | cut -d "," -f4)
  criterion=$(echo "$row" | cut -d "," -f5)
  optimizer=$(echo "$row" | cut -d "," -f6)
  epochs=$(echo "$row" | cut -d "," -f7)
  batch_size=$(echo "$row" | cut -d "," -f8)
  sampler=$(echo "$row" | cut -d "," -f9)
  augmentation=$(echo "$row" | cut -d "," -f10)
  input_channel=$(echo "$row" | cut -d "," -f11)
  echo "$i/$total: Training starts..."

  echo ""

  # Traning
  echo "$python $train_code --task $task --csv_name $csv_name --image_dir $image_dir --model $model --criterion $criterion --optimizer $optimizer --epochs $epochs --batch_size $batch_size --sampler $sampler --augmentation $augmentation --input_channel $input_channel --gpu_ids $gpu_ids"
  "$python" "$train_code" --task "$task" --csv_name "$csv_name" --image_dir "$image_dir" --model "$model" --criterion "$criterion" --optimizer "$optimizer" --epochs "$epochs" --batch_size "$batch_size" --sampler "$sampler" --augmentation "$augmentation" --input_channel "$input_channel" --gpu_ids "$gpu_ids" 2>&1 | tee -a "$train_log"

  echo ""

  # Test
  echo "$i/$total: Test starts..."
  echo "$python $test_code"
  "$python" "$test_code" 2>&1 | tee -a "$test_log"

  echo ""

  if [ "$task" = "classification" ]; then
    echo "$i/$total: Plot ROC..."
    echo "$python $roc_code"
    "$python" "$roc_code" 2>&1 | tee -a "$roc_log"
  elif [ "$task" = "regression" ]; then
    echo "$i/$total: Plot yy-graph..."
    echo "python $yy_code"
    "$python" "$yy_code" 2>&1 | tee -a "$yy_log"
  else
    # deepsurv
    echo "$i/$total: Calculate c-index..."
    echo "python $c_index_code"
    "$python" "$c_index_code" 2>&1 | tee -a "$c_index_log"
  fi

  i=$(($i + 1))
  echo -e "\n"
done
