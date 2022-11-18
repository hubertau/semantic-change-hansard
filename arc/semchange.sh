#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --job-name=sem_retrofit
#SBATCH --clusters=arc
#SBATCH --mem-per-cpu=350G
#SBATCH --output=/home/ball4321/2022-08a-Semantic_Change/logs/sem_%A.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk

module load Anaconda3//2020.11

source activate $DATA/venv_semantic

cd $HOME/2022-08a-Semantic_Change/src/semchange/

country=$1
lang=$2
event="911"
model=$3
split_date="2001-09-11"

echo $country
echo $lang
echo $event
echo $model
echo $split_date

python semchange.py \
  -f $DATA/data_semantic/01_raw/${country}_${lang}.csv \
  -c $DATA/data_semantic/02_intermediate/change_${event}_${lang}.txt \
  -nc $DATA/data_semantic/02_intermediate/nochange_${lang}.txt \
  --split_date ${split_date} \
  --split_range 3 \
  --model ${model} \
  --retrofit_outdir $DATA/data_semantic/02_intermediate/${country}_${lang}_${event}/ \
  --min_vocab_size 5000 \
  --tokenized_outdir $DATA/data_semantic/02_intermediate/${country}_${lang}_${event}/ \
  --outdir $DATA/data_semantic/04_models/${model}/${country}_${lang}_${event}/ \
  --model_output_dir $DATA/data_semantic/05_model_outputs/${model}/${country}_${lang}_${event}/ \
  --overwrite_preprocess \
  --overwrite_postprocess \
  --overwrite_model \
  # --log_level DEBUG \