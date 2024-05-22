#!/bin/bash

python3 src/semchange/semchange.py \
  -f data/01_raw/Corp_HouseOfCommons_V2.csv \
  -c data/02_intermediate/Corp_HouseOfCommons_change.txt \
  -nc data/02_intermediate/Corp_HouseOfCommons_nochange.txt \
  --outdir data/04_models/party_time_models \
  --model partytime