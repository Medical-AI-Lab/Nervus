#!/usr/bin/env bash

set -eu

parameter_csv="./parameter.csv"

train_log="./logger/logs/train.log"
test_log="./logger/logs/test.log"
eval_log="./logger/logs/eval.log"

#python="python3"
python="python"

train_code="train.py"
test_code="test.py"
eval_code="eval.py"

# Delete previous logs.
rm -f "$train_log"
rm -f "$test_log"
rm -f "$eval_log"


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
#11 in_channel,
#12 save_weight,
#13 gpu_ids
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
  in_channel=$(echo "$row" | cut -d "," -f11)
  save_weight=$(echo "$row" | cut -d "," -f12)
  gpu_ids=$(echo "$row" | cut -d "," -f13)

  echo "$i/$total: Training starts..."
  echo ""

  # Traning
  echo "$python $train_code --task $task --csv_name $csv_name --image_dir $image_dir --model $model --criterion $criterion --optimizer $optimizer --epochs $epochs --batch_size $batch_size --sampler $sampler --augmentation $augmentation --in_channel $in_channel --save_weight $save_weight --gpu_ids $gpu_ids"
  "$python" "$train_code" --task "$task" --csv_name "$csv_name" --image_dir "$image_dir" --model "$model" --criterion "$criterion" --optimizer "$optimizer" --epochs "$epochs" --batch_size "$batch_size" --sampler "$sampler" --augmentation "$augmentation" --in_channel "$in_channel" --save_weight "$save_weight" --gpu_ids "$gpu_ids" 2>&1 | tee -a "$train_log"

  echo ""

  # Test
  echo "$i/$total: Test starts..."
  echo "$python $test_code"
  "$python" "$test_code" 2>&1 | tee -a "$test_log"

  echo ""

  # Evaluation
  echo "$i/$total: Evaluation starts..."
  echo "$python $eval_code"
  "$python" "$eval_code" 2>&1 | tee -a "$eval_log"

  i=$(($i + 1))
  echo -e ""
done

echo "work_all done."