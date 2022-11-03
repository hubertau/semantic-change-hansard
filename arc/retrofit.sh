#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --job-name=sem_retrofit
#SBATCH --clusters=arc
#SBATCH --mem-per-cpu=350G
#SBATCH --output=../logs/sem_retrofit_%A.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hubert.au@oii.ox.ac.uk

module load Anaconda3//2020.11

source activate $DATA/venv_semantic

cd $HOME/2022-08a-Semantic_Change/src/semchange/

python semchange.py \
  -f $DATA/data_semantic/01_raw/Corp_HouseOfCommons_V2.csv \
  -c $DATA/data_semantic/02_intermediate/Corp_HouseOfCommons_change.txt \
  -nc $DATA/data_semantic/02_intermediate/Corp_HouseOfCommons_nochange.txt \
  --retrofit_outdir $DATA/data_semantic/02_intermediate/ \
  --outdir $DATA/data_semantic/04_models/aligned_for_retrofit/ \
  --model_output_dir $DATA/data_semantic/05_model_outputs \
  --model retrofit \
  --tokenized_outdir $DATA/data_semantic/02_intermediate