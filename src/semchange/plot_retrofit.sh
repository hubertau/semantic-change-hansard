CODE="uk_en_brexit"
# factor1="party-time-debate"
factor2="party-time"


# python plot_retrofit.py \
#   --retrofit_outfile /Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/05_model_outputs/retrofit/${CODE}/retrofit_out_${factor1}.txt \
#   --outdir /Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/${CODE}/${factor1}/

python plot_retrofit.py \
  --retrofit_outfile /Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/05_model_outputs/retrofit/${CODE}/retrofit_out_${factor2}.txt \
  --outdir /Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/${CODE}/${factor2}/