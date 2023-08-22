#!/bin/bash

python3 src/semchange/semchange.py \
  -f data/01_raw/Corp_HouseOfCommons_V2.csv \
  -c data/02_intermediate/Corp_HouseOfCommons_change.txt \
  -nc data/02_intermediate/Corp_HouseOfCommons_nochange.txt \
  --outdir data/04_models/mp_models \
  --model_output_dir data/05_model_outputs \
  --model speaker \
  --tokenized_outdir data/02_intermediate