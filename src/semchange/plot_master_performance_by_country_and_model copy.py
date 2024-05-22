import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():

    master_table_file = '/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/master_table.csv'
    master_table = pd.read_csv(master_table_file)

    for c in master_table['Country'].unique():
        # omit rows that had no data
        f = plt.figure(figsize=(15,8))
        sns.barplot(
            x = 'Model',
            y = 'Accuracy',
            hue= 'Logreg Type',
            data=master_table[(master_table['Fail'] > 0) & (master_table['Country'] == c)]
        )
        f.savefig(f'../../data/06_graphs/accuracy_by_model_{c}.png', bbox_inches='tight', dpi=800)

    for c in master_table['Country'].unique():
        # omit rows that had no data
        f = plt.figure(figsize=(15,8))
        sns.barplot(
            x = 'Model',
            y = 'Precision',
            hue= 'Logreg Type',
            data=master_table[(master_table['Fail'] > 0) & (master_table['Country'] == c)]
        )
        f.savefig(f'../../data/06_graphs/precision_by_model_{c}.png', bbox_inches='tight', dpi=800)

    for c in master_table['Country'].unique():
        # omit rows that had no data
        f = plt.figure(figsize=(15,8))
        sns.barplot(
            x = 'Model',
            y = 'Recall',
            hue= 'Logreg Type',
            data=master_table[(master_table['Fail'] > 0) & (master_table['Country'] == c)]
        )
        f.savefig(f'../../data/06_graphs/recall_by_model_{c}.png', bbox_inches='tight', dpi=800)

if __name__ == '__main__':
    main()