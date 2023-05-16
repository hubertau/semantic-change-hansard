import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():

    master_table_file = '/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/master_table.csv'
    master_table = pd.read_csv(master_table_file)

    # omit rows that had no data
    f = plt.figure(figsize=(15,8))
    sns.boxplot(
        x = 'Model',
        y = 'Recall',
        hue= 'Logreg Type',
        data=master_table[master_table['Fail'] > 0]
    )
    f.savefig('../../data/06_graphs/model_vs_recall.png', bbox_inches='tight', dpi=800)


if __name__ == '__main__':
    main()