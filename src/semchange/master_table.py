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
    files_logreg = glob.glob(os.path.join(scan_dir, '**', '*logreg.csv'), recursive=True)
    files_nn     = glob.glob(os.path.join(scan_dir, '**', '*nn_comparison.csv'), recursive=True)
    rows = []
    for f in files_logreg + files_nn:
        split_path = os.path.split(f)
        if 'logreg' in split_path[-1]:
            model_type = 'logreg'
        else:
            model_type = 'nn'
        country, lang, event = split_path[-2].split('_')
        data = pd.read_csv(f)
        model = data['Model'].iat[0]
        accuracy = data['Accuracy'].iat[0]
        precision = data['Precision'].iat[0]
        recall = data['Recall'].iat[0]
        f1_score = data['F1 Score'].iat[0]

        rows.append({
            "Model": model,
            "Type": model_type,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1_score,
            "Country": country,
            "Language": lang,
            "Event": event
        })

    mastertable = pd.DataFrame.from_records(rows)
    mastertable.to_csv(outfile)


if __name__ == '__main__':
    main()