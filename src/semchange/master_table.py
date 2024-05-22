"""Script to generate the new master table fromn ARC output.
"""

import pandas as pd
import numpy as np
import glob
import os
import click

@click.command()
@click.option('--scan_dir', required=True, help='directory within which to scan')
@click.option('--outfile', '-o', required=True, help='where to save the master table')
def main(scan_dir, outfile):
    files_logreg = glob.glob(os.path.join(scan_dir, '**', '*logreg*.csv'), recursive=True)
    # drop files with 'df' in them, those are diagnostic
    files_logreg = [i for i in files_logreg if 'df' not in os.path.split(i)[-1]]
    files_nn     = glob.glob(os.path.join(scan_dir, '**', '*nn_comparison.csv'), recursive=True)
    rows = []
    for f in files_logreg + files_nn:
        split_path = os.path.split(f)
        if 'logreg' in split_path[-1]:
            model_type = 'logreg'
        else:
            model_type = 'nn' 

        if 'retrofit' in f and 'logreg' in split_path[-1]:
            factor = split_path[1].split('logreg_')[1].split('.csv')[0]
        else:
            factor = None
        country, lang, event = split_path[0].split('/')[-1].split('_')
        data = pd.read_csv(f)
        for row in data.itertuples():
            try:
                model = row.Model
                accuracy = row.Accuracy
                precision = row.Precision
                recall = row.Recall
                f1_score = row.F1Score
                logreg_type = row.Logreg_type
                input_size = row._9
                change_count = row._10
                no_change_count = row._11


                rows.append({
                    "Model": model,
                    "Factor": factor,
                    "Type": model_type,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1_score,
                    "Country": country,
                    "Language": lang,
                    "Event": event,
                    "Logreg Type": logreg_type,
                    'Input_Size': input_size,
                    'Train_Change_Count': change_count,
                    'Train_No_Change_Count': no_change_count
                })
            except:
                print(row, f)

    mastertable = pd.DataFrame.from_records(rows)
    mastertable.to_csv(outfile)


if __name__ == '__main__':
    main()