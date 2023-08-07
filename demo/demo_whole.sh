#!/bin/bash
cd ../src/semchange

country=uk
lang=en
event="demo"
model=whole
# split_date="2001-09-11"
split_date="2008-10-01"
retrofit_factor="party-time"
req=0.3

echo $country
echo $lang
echo $event
echo $model
echo $split_date
echo $retrofit_factor

model_path=$model
if [[ "$model" == *"_plus"* ]]; then
  model_path=${model%"_plus"}
  echo $model_path
fi

python semchange.py \
  -f ../../data/01_raw/${country}_${lang}.csv \
  -c ../../demo/demo_change.txt \
  -nc ../../data/02_intermediate/nochange_${lang}.txt \
  --split_date ${split_date} \
  --split_range 5 \
  --model ${model} \
  --overlap_req ${req} \
  --retrofit_outdir ../../data/02_intermediate/${country}_${lang}_${event}/ \
  --retrofit_factor ${retrofit_factor} \
  --min_vocab_size 1000 \
  --tokenized_outdir ../../data/02_intermediate/${country}_${lang}_${event}/ \
  --outdir ../../data/04_models/${model_path}/${country}_${lang}_${event}/ \
  --model_output_dir ../../data/05_model_outputs/${model_path}/${country}_${lang}_${event}/ \
  --log_level INFO \
  # --overwrite_postprocess \
  # --overwrite_model \
  # --overwrite_preprocess \
  # --skip_model_check \