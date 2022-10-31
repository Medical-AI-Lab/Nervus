#!/usr/bin/env bash

set -eu

#parameter_csv="./parameter.csv"

python="python"

train_code="train.py"
test_code="test.py"
eval_code="eval.py"

while getopts ":g" optKey; do
  case "$optKey" in
    g)
      gpu_ids="0"
      ;;
  esac
done

#1  task
#2  csvpath
#3  model
#4  criterion
#5  optimizer
#6  epochs
#7  batch_size
#8  sampler
#9  augmtntation
#10 in_channel
#11 save_weight_policy
#12 gpu_ids
total=$(tail -n +1 "$parameter_csv" | wc -l)
i=1
for row in $(tail -n +2 "$parameter_csv"); do
  task=$(echo "$row" | cut -d "," -f1)
  csvpath=$(echo "$row" | cut -d "," -f2)
  model=$(echo "$row" | cut -d "," -f3)
  criterion=$(echo "$row" | cut -d "," -f4)
  optimizer=$(echo "$row" | cut -d "," -f5)
  epochs=$(echo "$row" | cut -d "," -f6)
  batch_size=$(echo "$row" | cut -d "," -f7)
  sampler=$(echo "$row" | cut -d "," -f8)
  augmentation=$(echo "$row" | cut -d "," -f9)
  in_channel=$(echo "$row" | cut -d "," -f10)
  save_weight_policy=$(echo "$row" | cut -d "," -f11)
  gpu_ids=$(echo "$row" | cut -d "," -f12)

  echo "$i/$total: Training starts..."
  echo ""

  # Traning
  echo "$python $train_code --task $task --csvpath $csvpath  --model $model --criterion $criterion --optimizer $optimizer --epochs $epochs --batch_size $batch_size --sampler $sampler --augmentation $augmentation --in_channel $in_channel --save_weight_policy $save_weight_policy --gpu_ids $gpu_ids"
  "$python" "$train_code" --task "$task" --csvpath "$csvpath" --model "$model" --criterion "$criterion" --optimizer "$optimizer" --epochs "$epochs" --batch_size "$batch_size" --sampler "$sampler" --augmentation "$augmentation" --in_channel "$in_channel" --save_weight_policy "$save_weight_policy" --gpu_ids "$gpu_ids"

  echo ""

  # Internal Test
  echo "$i/$total: Test starts..."
  echo "$python $test_code --csvpath $csvpath"
  "$python" "$test_code" --csvpath "$csvpath"

  echo ""

  # Evaluation
  echo "$i/$total: Evaluation starts..."
  echo "$python $eval_code"
  "$python" "$eval_code"

echo ""

  i=$(($i + 1))
  echo -e ""
done

echo "work_all done."
