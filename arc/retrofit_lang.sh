#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --job-name=sem_retrofit
#SBATCH --clusters=arc
#SBATCH --mem-per-cpu=350G
#SBATCH --output=/home/ball4321/2022-08a-Semantic_Change/logs/sem_retrofit_es_%A.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk

module load Anaconda3//2020.11

source activate $DATA/venv_semantic

cd $HOME/2022-08a-Semantic_Change/src/semchange/

lang=$1
country=$2
event="911"
model="retrofit"
split_date="2001-09-11 23:59:59"

python semchange.py \
  -f $DATA/data_semantic/01_raw/${lang}_${country}.csv \
  -c $DATA/data_semantic/02_intermediate/change_911_${lang}.txt \
  -nc $DATA/data_semantic/02_intermediate/nochange_${lang}.txt \
  --split_date ${split_date} \
  --model ${model} \
  --retrofit_outdir $DATA/data_semantic/02_intermediate/${lang}_${country}_${event}/ \
  --outdir $DATA/data_semantic/04_models/retrofit/${lang}_${country}_${event}/ \
  --model_output_dir $DATA/data_semantic/05_model_outputs \
  --tokenized_outdir $DATA/data_semantic/02_intermediate/${lang}_${country}_${event}/