import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import gensim
import umap.umap_ as umap
import numpy as np
import os
import pandas as pd

from semchange import synonym_item

import click

@click.command()
@click.option('--retrofit_outfile', required=True)
@click.option('--outdir', required=True)
def main(retrofit_outfile, outdir):
    print('Initialisaing UMAP')
    reducer = umap.UMAP(random_state=999, n_neighbors=30, min_dist=.25)

    print('Reading in vectors')
    with open(retrofit_outfile) as f:
        vecs = f.readlines()
        vecs = [vec.replace('\n', '')for vec in vecs]
    vec_length=300

    outmatrix = np.zeros(shape=(len(vecs), vec_length))
    index_to_key = []
    for i in range(len(vecs)):

        vec = vecs[i].strip().split(' ')
        # Extracting synonym key
        index_to_key.append(vec[0])
        del(vec[0])
        vec=[i for i in vec[1:] if i!='']
        vec =[float(v) for v in vec]
        assert len(vec) == vec_length
        outmatrix[i, :] = np.array(vec)

    # reduce
    print('Doing UMAP')
    embedding = pd.DataFrame(reducer.fit_transform(outmatrix), columns = ['UMAP1','UMAP2'])
    print('Creating DataFrame')
    embedding['synkey'] = index_to_key
    temp = embedding['synkey'].apply(synonym_item.from_string)
    embedding['word']    = temp.apply(lambda x: x.word)
    embedding['speaker'] = temp.apply(lambda x: x.speaker)
    embedding['party']   = temp.apply(lambda x: x.party)
    embedding['time']    = temp.apply(lambda x: x.time)
    embedding.to_csv(os.path.join(outdir, 'embedding.csv'))

    # plot, colour by party
    # fig, ax = plt.figure(figsize=(15,8))
    print('Plotting')
    sns.set_theme()
    for word in embedding['word'].unique():
        # try:
        sns_plot = plt.figure()
        # sns_plot = sns.jointplot(
        #     x='UMAP1',
        #     y='UMAP2',
        #     data=embedding[(embedding['word']==word) & (embedding['time']=='t1')],
        #     hue='party',
        #     label='t1'
        #     # alpha=0.9,
        #     # style='time'
        # )
        # data_t2=embedding[(embedding['word']==word) & (embedding['time']=='t2')]
        # sns_plot.x = data_t2.UMAP1
        # sns_plot.y = data_t2.UMAP2
        # sns_plot.plot_joint(
        #     sns.scatterplot,
        #     hue='party',
        #     markers=['x'],
        #     label='t2'
        # )
        # ax = plt.gca()
        # ax.legend()  #CHANGE HERE
        sns_plot = sns.scatterplot(
            x='UMAP1',
            y='UMAP2',
            data = embedding[embedding['word']==word],
            hue='party',
            style='time'
        )
        sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
        # sns_plot.plot_marginals(sns.kdeplot, color='b', shade=True, alpha=.2, legend=False)
        graph_savepath = f'plot_{word.upper()}_by_PARTY_{os.path.split(retrofit_outfile)[-1].split(".")[0]}.png'
        sns_plot.figure.savefig(os.path.join(outdir, graph_savepath), bbox_inches='tight', dpi=500)
        plt.close()
        # except:
            # print(embedding[embedding['word']==word])


if __name__ == '__main__':
    main()