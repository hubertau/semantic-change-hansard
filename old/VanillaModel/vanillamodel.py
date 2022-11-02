# %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:18:41.965160Z","iopub.execute_input":"2022-06-22T09:18:41.965896Z","iopub.status.idle":"2022-06-22T09:18:53.213467Z","shell.execute_reply.started":"2022-06-22T09:18:41.965773Z","shell.execute_reply":"2022-06-22T09:18:53.212650Z"}}
#!pip install researchpy

# %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:18:53.215843Z","iopub.execute_input":"2022-06-22T09:18:53.216193Z","iopub.status.idle":"2022-06-22T09:18:55.882263Z","shell.execute_reply.started":"2022-06-22T09:18:53.216148Z","shell.execute_reply":"2022-06-22T09:18:55.881411Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import gensim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import csv
from csv import reader
from scipy import spatial
import functools
from collections import Counter
import nltk
from nltk.data import load

import imblearn
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

import researchpy as rp
import scipy.stats as stats

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:18:55.883384Z","iopub.execute_input":"2022-06-22T09:18:55.883710Z","iopub.status.idle":"2022-06-22T09:19:22.083482Z","shell.execute_reply.started":"2022-06-22T09:18:55.883680Z","shell.execute_reply":"2022-06-22T09:19:22.082912Z"}}
#%%time
# open file in read mode
#speechesPath = '.\Resources\utf8tokenizedspeeches\TokenizedSpeeches_utf-8.csv'
def vanilla():

    speechesPath = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\TokenizedSpeeches_utf-8.csv"
    #with open(r".\Resources\utf8tokenizedspeeches\TokenizedSpeeches_utf-8.csv", 'r') as read_obj:
    #speechesPath, 'r') as read_obj:

    with open(speechesPath, 'r') as read_obj:

        lemmasList = []
    
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader: 
            lemmasList.append(row)
        print(len(lemmasList), 'Rows read')
    

    #create dataframe from the lemmas extracted from csv
    dictOfLemmas = {'Lemmas': lemmasList}
    lemmasDf = pd.DataFrame(dictOfLemmas)

    speechesPklPath = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\hansard-speeches-post2010.pkl"
    #with open('/kaggle/input/hansard-speeches-lemmatized/hansard-speeches-post2010.pkl', 'rb') as f:
    with open(speechesPklPath, 'rb') as f:

        df = pickle.load(f)
    
    #since index was missing values and didn't match with the lemmasDf index
    df = df.reset_index(drop=True)
    df = df.join(lemmasDf)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:22.085256Z","iopub.execute_input":"2022-06-22T09:19:22.085665Z","iopub.status.idle":"2022-06-22T09:19:22.456878Z","shell.execute_reply.started":"2022-06-22T09:19:22.085634Z","shell.execute_reply":"2022-06-22T09:19:22.456267Z"}}
    # Split data based on the Brexit referendum event before and after period
    eventDate = '2016-06-23 23:59:59'
    df_t1 = df[df['date']<= eventDate]
    df_t2 = df[df['date']> eventDate]

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:22.458257Z","iopub.execute_input":"2022-06-22T09:19:22.458728Z","iopub.status.idle":"2022-06-22T09:19:29.457641Z","shell.execute_reply.started":"2022-06-22T09:19:22.458687Z","shell.execute_reply":"2022-06-22T09:19:29.456770Z"}}

    model1Path = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\models12\model1"
    model2Path = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\models12\model2"

    '''model1 = gensim.models.Word2Vec.load('../input/models12/model1')
    model2 = gensim.models.Word2Vec.load('../input/models12/model2')'''


    model1 = gensim.models.Word2Vec.load(model1Path)
    model2 = gensim.models.Word2Vec.load(model2Path)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:29.459572Z","iopub.execute_input":"2022-06-22T09:19:29.459784Z","iopub.status.idle":"2022-06-22T09:19:29.484417Z","shell.execute_reply.started":"2022-06-22T09:19:29.459757Z","shell.execute_reply":"2022-06-22T09:19:29.483499Z"}}
    def intersection_align_gensim(m1, m2, words=None):
        """
        Intersect two gensim word2vec models, m1 and m2.
        Only the shared vocabulary between them is kept.
        If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
        Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
        These indices correspond to the new syn0 and syn0norm objects in both gensim models:
            -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
            -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
        The .vocab dictionary is also updated for each model, preserving the count but updating the index.
        """
        # Get the vocab for each model
        vocab_m1 = set(m1.wv.index_to_key)
        vocab_m2 = set(m2.wv.index_to_key)

        # Find the common vocabulary
        common_vocab = vocab_m1 & vocab_m2
        if words: common_vocab &= set(words)
        
        # If no alignment necessary because vocab is identical...
        if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
            return (m1,m2)

        # Otherwise sort by frequency (summed for both)
        common_vocab = list(common_vocab)
        common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
        # Then for each model...
        for m in [m1, m2]:
            # Replace old syn0norm array with new one (with common vocab)
            indices = [m.wv.key_to_index[w] for w in common_vocab]
            old_arr = m.wv.vectors
            new_arr = np.array([old_arr[index] for index in indices])
            m.wv.vectors = new_arr

            # Replace old vocab dictionary with new one (with common vocab)
            # and old index2word with new one
            new_key_to_index = {}
            new_index_to_key = []
            for new_index, key in enumerate(common_vocab):
                new_key_to_index[key] = new_index
                new_index_to_key.append(key)
            m.wv.key_to_index = new_key_to_index
            m.wv.index_to_key = new_index_to_key
        
        print('Vocab function returning models with shapes', m1.wv.vectors.shape, m2.wv.vectors.shape)    
        return (m1,m2)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:29.486198Z","iopub.execute_input":"2022-06-22T09:19:29.486524Z","iopub.status.idle":"2022-06-22T09:19:29.505499Z","shell.execute_reply.started":"2022-06-22T09:19:29.486482Z","shell.execute_reply":"2022-06-22T09:19:29.504546Z"}}
    # Function to align two spaces with orthogunal procrustes
    def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
        print('shapes', base_embed.wv.vectors.shape, other_embed.wv.vectors.shape)
        """
        Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
        Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
        Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
        First, intersect the vocabularies (see `intersection_align_gensim` documentation).
        Then do the alignment on the other_embed model.
        Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
        Return other_embed.
        If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
        """

        # make sure vocabulary and indices are aligned
        in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

        # re-filling the normed vectors
        in_base_embed.wv.fill_norms(force=True)
        in_other_embed.wv.fill_norms(force=True)
    
        # get the (normalized) embedding matrices
        base_vecs = in_base_embed.wv.get_normed_vectors()
    
        other_vecs = in_other_embed.wv.get_normed_vectors()
        # just a matrix dot product with numpy
        m = other_vecs.T.dot(base_vecs) 

        # SVD method from numpy
        u, _, v = np.linalg.svd(m)

        # another matrix operation
        ortho = u.dot(v) 

        # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
        other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
        print('Procrustes function returning')
        return other_embed

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:29.507012Z","iopub.execute_input":"2022-06-22T09:19:29.507305Z","iopub.status.idle":"2022-06-22T09:19:30.461779Z","shell.execute_reply.started":"2022-06-22T09:19:29.507265Z","shell.execute_reply":"2022-06-22T09:19:30.460788Z"}}
    # Applying the functions to our models

    smart_procrustes_align_gensim(model1, model2, words=None)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:30.467309Z","iopub.execute_input":"2022-06-22T09:19:30.467882Z","iopub.status.idle":"2022-06-22T09:19:30.477800Z","shell.execute_reply.started":"2022-06-22T09:19:30.467826Z","shell.execute_reply":"2022-06-22T09:19:30.474924Z"}}
    def cosine_similarity(word):
      sc = 1-spatial.distance.cosine(model1.wv[word], model2.wv[word])
      return sc

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:30.481463Z","iopub.execute_input":"2022-06-22T09:19:30.487141Z","iopub.status.idle":"2022-06-22T09:19:35.908017Z","shell.execute_reply.started":"2022-06-22T09:19:30.487071Z","shell.execute_reply":"2022-06-22T09:19:35.907108Z"}}
    cosine_similarity_df = pd.DataFrame(([w, cosine_similarity(w), model1.wv.get_vecattr(w, "count") , model2.wv.get_vecattr(w, "count") ] for w in model1.wv.index_to_key), columns = ('Word', 'Cosine_similarity', "Frequency_t1", "Frequency_t2"))

    cosine_similarity_df['FrequencyRatio'] = cosine_similarity_df['Frequency_t1']/cosine_similarity_df['Frequency_t2']
    cosine_similarity_df['TotalFrequency'] = cosine_similarity_df['Frequency_t1'] + cosine_similarity_df['Frequency_t2']

    cosine_similarity_df_sorted = cosine_similarity_df.sort_values(by='Cosine_similarity', ascending=True)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:35.909414Z","iopub.execute_input":"2022-06-22T09:19:35.909651Z","iopub.status.idle":"2022-06-22T09:19:35.923025Z","shell.execute_reply.started":"2022-06-22T09:19:35.909621Z","shell.execute_reply":"2022-06-22T09:19:35.921414Z"}}
    change = ['exiting', 'seaborne', 'eurotunnel', 'withdrawal', 'departures', 'unicorn', 'remainers', 'exit', 'surrender',
              'departure', 'triggering', 'stockpiling', 'expulsion', 'blindfold', 'cliff', 'lighter', 'exits', 'triggered',
              'brexiteer', 'soft', 'plus', 'trigger', 'backroom', 'invoked', 'protesting', 'brexit', 'edge', 'canary', 
              'unicorns', 'withdrawing', 'invoking', 'withdrawn', 'manor', 'brexiteers', 'fanatics', 'postponement', 
              'currencies', 'currency', 'operability', 'operable', 'leavers', 'invoke', 'article', 'eurozone', 'clueless',
              'surrendered', 'cake', 'red', 'euroscepticism', 'prorogation', 'lining', 'gove', 'norway', 'deflationary',
              'moribund', 'eurosceptic', 'deutschmark', 'courting', 'deal', 'withdraw', 'dab', 'withdrawals', 'eurosceptics',
              'surrendering', 'aldous', 'lanarkshire', 'leaving', 'signifying', 'roofs', 'ceded', 'absentia', 'treachery',
              'dollar', 'canada', 'pragmatist', 'oven', 'ready', 'brexiters', 'control', 'capitulation', 'leave', 'referendum',
              'agreement', 'prorogue', 'smoothest', 'depreciate', 'managed', 'mutiny', 'overvalued', 'ideologues', 'foreign',
              'eec', 'war', 'prorogued', 'hannan', 'appease', 'pendolino', 'southbound', 'left', 'line', 'hard', 'bill']
 
    no_change = ['prime', 'even', 'parliament', 'care', 'well', 'constituency', 'tax', 'children',
                 'business', 'report', 'case', 'sure', 'like', 'see', 'state', 'order', 'back', 'new', 'hope', 'local',
                 'secretary', 'public', 'right', 'much', 'say', 'first', 'minister', 'look', 'system', 'whether', 
                 'members', 'million', 'good', 'today', 'services', 'clear', 'help', 'time', 'place', 'put', 'last', 'must', 'money', 'one', 
                 'way', 'work', 'would', 'think', 'two', 'great', 'could', 'lady', 'us', 'come', 'however', 'may', 'going', 'go',
                 'given', 'year', 'might', 'part', 'get', 'make', 'point', 'committee', 'years', 'also', 'know',
                 'government', 'take', 'house', 'agree', 'member', 'number', 'across', 'made', 'give', 'gentleman', 'important', 'said',
                 'people', 'issue', 'support', 'ensure']


    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:35.924492Z","iopub.execute_input":"2022-06-22T09:19:35.924884Z","iopub.status.idle":"2022-06-22T09:19:35.966531Z","shell.execute_reply.started":"2022-06-22T09:19:35.924833Z","shell.execute_reply":"2022-06-22T09:19:35.965857Z"}}
    words_of_interest = cosine_similarity_df_sorted[cosine_similarity_df_sorted['Word'].isin(change+no_change)]

    words_of_interest.loc[words_of_interest['Word'].isin(change), 'semanticDifference'] = 'change' 
    words_of_interest.loc[words_of_interest['Word'].isin(no_change), 'semanticDifference'] = 'no_change' 

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:35.967779Z","iopub.execute_input":"2022-06-22T09:19:35.968400Z","iopub.status.idle":"2022-06-22T09:19:35.975142Z","shell.execute_reply.started":"2022-06-22T09:19:35.968348Z","shell.execute_reply":"2022-06-22T09:19:35.974396Z"}}
    change_cossim = words_of_interest.loc[words_of_interest['semanticDifference'] == 'change', 'Cosine_similarity'] 
    no_change_cossim = words_of_interest.loc[words_of_interest['semanticDifference'] == 'no_change', 'Cosine_similarity'] 

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:35.976325Z","iopub.execute_input":"2022-06-22T09:19:35.976691Z","iopub.status.idle":"2022-06-22T09:19:36.215778Z","shell.execute_reply.started":"2022-06-22T09:19:35.976655Z","shell.execute_reply":"2022-06-22T09:19:36.214929Z"}}
    change_cossim.hist()

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:36.217117Z","iopub.execute_input":"2022-06-22T09:19:36.217420Z","iopub.status.idle":"2022-06-22T09:19:36.448045Z","shell.execute_reply.started":"2022-06-22T09:19:36.217377Z","shell.execute_reply":"2022-06-22T09:19:36.447227Z"}}
    no_change_cossim.hist()

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:36.449162Z","iopub.execute_input":"2022-06-22T09:19:36.449405Z","iopub.status.idle":"2022-06-22T09:19:36.683963Z","shell.execute_reply.started":"2022-06-22T09:19:36.449375Z","shell.execute_reply":"2022-06-22T09:19:36.683346Z"}}
    words_of_interest['Cosine_similarity'].hist()

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:19:36.685279Z","iopub.execute_input":"2022-06-22T09:19:36.685687Z","iopub.status.idle":"2022-06-22T09:19:36.693938Z","shell.execute_reply.started":"2022-06-22T09:19:36.685656Z","shell.execute_reply":"2022-06-22T09:19:36.693010Z"}}
    words_of_interest['semanticDifference'].value_counts()

    # %% [markdown]
    # # **Logistic Regression**

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:23:56.262479Z","iopub.execute_input":"2022-06-22T09:23:56.264158Z","iopub.status.idle":"2022-06-22T09:23:56.296669Z","shell.execute_reply.started":"2022-06-22T09:23:56.264039Z","shell.execute_reply":"2022-06-22T09:23:56.295561Z"}}
    X = words_of_interest['Cosine_similarity'].values.reshape(-1,1)
    y = words_of_interest['semanticDifference']

    undersample = RandomUnderSampler(sampling_strategy=1.0)

    X_over, y_over = undersample.fit_resample(X, y)
    X, y = X_over, y_over

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    logreg = LogisticRegression()
    kf = logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    #print('Y value counts',y.value_counts(),'\n')
    #print('Y train value counts', y_train.value_counts())
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:00.825470Z","iopub.execute_input":"2022-06-22T09:25:00.825816Z","iopub.status.idle":"2022-06-22T09:25:00.922181Z","shell.execute_reply.started":"2022-06-22T09:25:00.825779Z","shell.execute_reply":"2022-06-22T09:25:00.921322Z"}}
    scoring = {'accuracy' : make_scorer(accuracy_score), 
                   'precision' : make_scorer(precision_score,pos_label='change'),
                   'recall' : make_scorer(recall_score,pos_label='change'), 
                   'f1_score' : make_scorer(f1_score,pos_label='change')}

    scores = cross_validate(kf, X, y, cv=10, scoring=scoring,error_score='raise')


    '''
    print('Accuracy', scores['test_accuracy'].mean())
    print('Precision', scores['test_precision'].mean())
    print('Recall', scores['test_recall'].mean())
    print('F1 Score', scores['test_f1_score'].mean())'''


    accuracy, precision, recall, f1_score = [], [], [], []

    accuracy.append(scores['test_accuracy'].mean())
    precision.append(scores['test_precision'].mean())
    recall.append(scores['test_recall'].mean())
    f1_score.append(scores['test_f1_score'].mean())

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:07.538416Z","iopub.execute_input":"2022-06-22T09:25:07.539419Z","iopub.status.idle":"2022-06-22T09:25:07.562441Z","shell.execute_reply.started":"2022-06-22T09:25:07.539371Z","shell.execute_reply":"2022-06-22T09:25:07.561844Z"}}
    scoresDict = {'Model':['Vanilla Model'], 'Description': ['Corpus split as -T1/T2'], 'Basis': ['Cosine Similarity'],'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1Score':f1_score}
    scoresDf = pd.DataFrame(scoresDict)
    print('Scores DF once created')

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    #from scipy import stats
    #stats.ttest_ind(group1, group2, equal_var=False)

    # %% [markdown]
    # # **Nearest Neighbours' comparison**
    print('NEAREST NEIGHBOURS 0')
    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:23.172661Z","iopub.execute_input":"2022-06-22T09:25:23.173368Z","iopub.status.idle":"2022-06-22T09:25:24.644446Z","shell.execute_reply.started":"2022-06-22T09:25:23.173324Z","shell.execute_reply":"2022-06-22T09:25:24.643449Z"}}
    neighboursInT1 = []
    neighboursInT2 = []

    for word in words_of_interest['Word'].to_list():
    
        x = model1.wv.similar_by_word(word,10) 
        y = model2.wv.similar_by_word(word,10)

        x = [tup[0] for tup in x]
        y = [tup[0] for tup in y]
    
        neighboursInT1.append(x)
        neighboursInT2.append(y)
    
    words_of_interest['neighboursInT1'] = neighboursInT1
    words_of_interest['neighboursInT2'] = neighboursInT2

    #words_of_interest['overlappingNeighbours'] = ?
    #intersectingNeighbs = set(words_of_interest['neighboursInT1'].to_list()).intersect(words_of_interest['neighboursInT2'].to_list())
    lengthOverlap = []

    for index in (words_of_interest['neighboursInT1'].index):
        neighboursT1 = words_of_interest.at[index, 'neighboursInT1']
        neighboursT2 = words_of_interest.at[index, 'neighboursInT2']
        lengthOverlap.append(len(set(neighboursT1).intersection(neighboursT2)))

    words_of_interest['overlappingNeighbours'] = lengthOverlap

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:31.874833Z","iopub.execute_input":"2022-06-22T09:25:31.875412Z","iopub.status.idle":"2022-06-22T09:25:31.892243Z","shell.execute_reply.started":"2022-06-22T09:25:31.875369Z","shell.execute_reply":"2022-06-22T09:25:31.891376Z"},"jupyter":{"outputs_hidden":true}}
    words_of_interest[words_of_interest['semanticDifference']=='change']['overlappingNeighbours'].describe()

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:32.298050Z","iopub.execute_input":"2022-06-22T09:25:32.298785Z","iopub.status.idle":"2022-06-22T09:25:32.308882Z","shell.execute_reply.started":"2022-06-22T09:25:32.298740Z","shell.execute_reply":"2022-06-22T09:25:32.308178Z"},"jupyter":{"outputs_hidden":true}}
    words_of_interest[words_of_interest['semanticDifference']=='no_change']['overlappingNeighbours'].describe()

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:42.047205Z","iopub.execute_input":"2022-06-22T09:25:42.047999Z","iopub.status.idle":"2022-06-22T09:25:42.057401Z","shell.execute_reply.started":"2022-06-22T09:25:42.047949Z","shell.execute_reply":"2022-06-22T09:25:42.056153Z"}}
    neighbours_of_changed_words = words_of_interest[words_of_interest['semanticDifference']=='change'].sort_values(by='Cosine_similarity',ascending=True)[['Word','neighboursInT1','neighboursInT2']]

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T08:49:53.121656Z","iopub.execute_input":"2022-06-22T08:49:53.123185Z","iopub.status.idle":"2022-06-22T08:49:53.127601Z","shell.execute_reply.started":"2022-06-22T08:49:53.123124Z","shell.execute_reply":"2022-06-22T08:49:53.126497Z"}}
    #out_path = './'
    #neighbours_of_changed_words.to_csv(os.path.join(out_path, 'neighbours_of_changed_words.csv'), encoding='utf-8', columns = ['Word', 'neighboursInT1', 'neighboursInT2'])

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:49.977517Z","iopub.execute_input":"2022-06-22T09:25:49.978254Z","iopub.status.idle":"2022-06-22T09:25:49.999773Z","shell.execute_reply.started":"2022-06-22T09:25:49.978205Z","shell.execute_reply":"2022-06-22T09:25:49.999087Z"}}
    X = words_of_interest['overlappingNeighbours'].values.reshape(-1,1)
    y = words_of_interest['semanticDifference']

    undersample = RandomUnderSampler(sampling_strategy=1.0)

    X_over, y_over = undersample.fit_resample(X, y)
    X=X_over
    y=y_over

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    logreg = LogisticRegression()
    kf = logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    #print('Y value counts',y.value_counts(),'\n')
    #print('Y train value counts', y_train.value_counts())
    print('NEAREST NEIGHBOURS')
    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:53.300313Z","iopub.execute_input":"2022-06-22T09:25:53.300744Z","iopub.status.idle":"2022-06-22T09:25:53.305345Z","shell.execute_reply.started":"2022-06-22T09:25:53.300692Z","shell.execute_reply":"2022-06-22T09:25:53.304762Z"}}
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:55.742717Z","iopub.execute_input":"2022-06-22T09:25:55.743499Z","iopub.status.idle":"2022-06-22T09:25:55.838760Z","shell.execute_reply.started":"2022-06-22T09:25:55.743461Z","shell.execute_reply":"2022-06-22T09:25:55.837895Z"}}
    scoring = {'accuracy' : make_scorer(accuracy_score), 
                   'precision' : make_scorer(precision_score,pos_label='change'),
                   'recall' : make_scorer(recall_score,pos_label='change'), 
                   'f1_score' : make_scorer(f1_score,pos_label='change')}

    scores = cross_validate(kf, X, y, cv=10, scoring=scoring,error_score='raise')
    accuracy, precision, recall, f1_score = [], [], [], []


    accuracy.append(scores['test_accuracy'].mean())
    precision.append(scores['test_precision'].mean())
    recall.append(scores['test_recall'].mean())
    f1_score.append(scores['test_f1_score'].mean())

    print('STUFF APPENDED')

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T09:25:58.541209Z","iopub.execute_input":"2022-06-22T09:25:58.541487Z","iopub.status.idle":"2022-06-22T09:25:58.558683Z","shell.execute_reply.started":"2022-06-22T09:25:58.541458Z","shell.execute_reply":"2022-06-22T09:25:58.558063Z"}}
    scoresDict = {'Model':['Vanilla Model'], 'Description': ['Corpus split as -T1/T2'], 'Basis': ['Overlap of nearest 10 neighbours'], 'Accuracy':accuracy, 'Precision':precision, 'Recall': recall, 'F1Score': f1_score}
    scoresDf = pd.concat([scoresDf, pd.DataFrame(scoresDict)])

    return scoresDf
