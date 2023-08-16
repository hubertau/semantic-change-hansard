import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/master_table.csv')

    pivoted = data.groupby(['Logreg Type', 'Model']).mean().reset_index().pivot('Logreg Type', 'Model', 'Accuracy')

    f = plt.figure(figsize=(15,8))
    sns.heatmap(pivoted, annot=True, fmt = ".2f")
    f.savefig('/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/master_heatmap.png', bbox_inches='tight', dpi=800)

    # by country
    for i in ['speaker',  'whole', 'retrofit', 'speaker_plus']:

        temp = data[data['Model'] == i]
        pivoted = temp.groupby(['Logreg Type', 'Country']).mean().reset_index().pivot('Logreg Type', 'Country', 'Accuracy')

        f = plt.figure(figsize=(15,8))
        sns.heatmap(pivoted, annot=True, fmt = ".2f")
        f.savefig(f'/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/master_heatmap_country_{i}.png', bbox_inches='tight', dpi=800)

    # F1 by country
    for i in ['speaker',  'whole', 'retrofit', 'speaker_plus']:

        temp = data[data['Model'] == i]
        pivoted = temp.groupby(['Logreg Type', 'Country']).mean().reset_index().pivot('Logreg Type', 'Country', 'F1')

        f = plt.figure(figsize=(15,8))
        sns.heatmap(pivoted, annot=True, fmt = ".2f")
        f.savefig(f'/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/06_graphs/master_heatmap_country_{i}_F1.png', bbox_inches='tight', dpi=800)


if __name__ == '__main__':
    main()