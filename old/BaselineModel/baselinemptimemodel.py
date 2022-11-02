# %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:55:46.069866Z","iopub.execute_input":"2022-06-25T20:55:46.070234Z","iopub.status.idle":"2022-06-25T20:55:59.289491Z","shell.execute_reply.started":"2022-06-25T20:55:46.070204Z","shell.execute_reply":"2022-06-25T20:55:59.288379Z"}}

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

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

import imblearn
from imblearn.under_sampling import RandomUnderSampler

import researchpy as rp
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Load aligned 24 Word2Vec models of MPs in T1 & T2

def mpTime():
    
    dictOfModels = {}
    folderPath = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\24-aligned-models-by-mp-and-time"

    #Loading aligned models 
    for file in os.listdir(folderPath):
        filePath = folderPath + '/' + file
        model = gensim.models.Word2Vec.load(filePath)
        dictOfModels[file] = model

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:51:21.405648Z","iopub.execute_input":"2022-06-25T20:51:21.406261Z","iopub.status.idle":"2022-06-25T20:51:21.420418Z","shell.execute_reply.started":"2022-06-25T20:51:21.406227Z","shell.execute_reply":"2022-06-25T20:51:21.419280Z"}}
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

    words_of_interest= change+no_change

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:51:21.907080Z","iopub.execute_input":"2022-06-25T20:51:21.907438Z","iopub.status.idle":"2022-06-25T20:51:21.917243Z","shell.execute_reply.started":"2022-06-25T20:51:21.907408Z","shell.execute_reply":"2022-06-25T20:51:21.915829Z"}}
    # Slightly modified to now find the cosine difference between provided vectors instead of
    # fetching vectors from known models 
    def cosine_similarity(vec1, vec2):
      sc = 1-spatial.distance.cosine(vec1, vec2)
      return sc

    cosine_similarity_df = pd.DataFrame(columns = ('Word', 'Cosine_similarity'))

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:51:22.887808Z","iopub.execute_input":"2022-06-25T20:51:22.888179Z","iopub.status.idle":"2022-06-25T20:51:22.896213Z","shell.execute_reply.started":"2022-06-25T20:51:22.888148Z","shell.execute_reply":"2022-06-25T20:51:22.894837Z"}}
    def computeAvgVec(mKeys, w):
        if(w in dictOfModels[mKeys[0]].wv.index_to_key):
            modelsSum = np.zeros(dictOfModels[mKeys[0]].layer1_size)
            for k in mKeys:
                vectorPerModel = dictOfModels[k].wv[w]
                modelsSum = np.add(modelsSum, vectorPerModel)
            avgEmbedding =np.divide(modelsSum, len(mKeys))
            return avgEmbedding
        else:
            print('Word '+str(w) + ' not found in models vocab')
            return []

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:51:23.552319Z","iopub.execute_input":"2022-06-25T20:51:23.552682Z","iopub.status.idle":"2022-06-25T20:51:23.558571Z","shell.execute_reply.started":"2022-06-25T20:51:23.552652Z","shell.execute_reply":"2022-06-25T20:51:23.557219Z"}}
    words_of_interest = change + no_change

    t1Keys = [k for k in dictOfModels.keys() if 'df_t1' in k] 
    t2Keys = [k for k in dictOfModels.keys() if 'df_t2' in k] 

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:51:28.451710Z","iopub.execute_input":"2022-06-25T20:51:28.452112Z","iopub.status.idle":"2022-06-25T20:51:28.737139Z","shell.execute_reply.started":"2022-06-25T20:51:28.452079Z","shell.execute_reply":"2022-06-25T20:51:28.735917Z"},"jupyter":{"outputs_hidden":true}}
    # Compute average of word in T1 and in T2 and store average vectors and cosine difference   
    for word in words_of_interest:
            print(word)
            #Provide a list of keys to average computation model for it to
            #compute average vector amongst these models

            avgVecT1 = computeAvgVec(t1Keys, word)
            avgVecT2 = computeAvgVec(t2Keys, word)

            if(avgVecT1==[] or avgVecT2==[]):

                print(str(word) + ' Word not found')
                continue

            else:

                # Cos similarity between averages
                cosSimilarity = cosine_similarity(avgVecT1, avgVecT2)
                cosine_similarity_df = cosine_similarity_df.append({'Word': word, 'Cosine_similarity': cosSimilarity}, ignore_index=True)


    words_of_interest = cosine_similarity_df[cosine_similarity_df['Word'].isin(change+no_change)]

    words_of_interest.loc[words_of_interest['Word'].isin(change), 'semanticDifference'] = 'change' 
    words_of_interest.loc[words_of_interest['Word'].isin(no_change), 'semanticDifference'] = 'no_change' 

    words_of_interest.sort_values(by='Cosine_similarity')

    # %% [markdown]
    # # **LOGISTIC REGRESSION**

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:53:52.542811Z","iopub.execute_input":"2022-06-25T20:53:52.543369Z","iopub.status.idle":"2022-06-25T20:53:52.549420Z","shell.execute_reply.started":"2022-06-25T20:53:52.543326Z","shell.execute_reply":"2022-06-25T20:53:52.548470Z"}}
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:55:59.291070Z","iopub.execute_input":"2022-06-25T20:55:59.292258Z","iopub.status.idle":"2022-06-25T20:55:59.323240Z","shell.execute_reply.started":"2022-06-25T20:55:59.292213Z","shell.execute_reply":"2022-06-25T20:55:59.322111Z"}}
    X = words_of_interest['Cosine_similarity'].values.reshape(-1,1)
    y = words_of_interest['semanticDifference']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

    undersample = RandomUnderSampler(sampling_strategy=1.0)

    X_over, y_over = undersample.fit_resample(X, y)
    X=X_over
    y=y_over

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    logreg = LogisticRegression()
    kf = logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:56:02.594343Z","iopub.execute_input":"2022-06-25T20:56:02.594713Z","iopub.status.idle":"2022-06-25T20:56:02.679675Z","shell.execute_reply.started":"2022-06-25T20:56:02.594683Z","shell.execute_reply":"2022-06-25T20:56:02.678372Z"}}
    scoring = {'accuracy' : make_scorer(accuracy_score), 
                   'precision' : make_scorer(precision_score,pos_label='change'),
                   'recall' : make_scorer(recall_score,pos_label='change'), 
                   'f1_score' : make_scorer(f1_score,pos_label='change')}

    scores = cross_validate(kf, X, y, cv=10, scoring=scoring,error_score='raise')

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:56:12.316450Z","iopub.execute_input":"2022-06-25T20:56:12.316837Z","iopub.status.idle":"2022-06-25T20:56:12.612188Z","shell.execute_reply.started":"2022-06-25T20:56:12.316806Z","shell.execute_reply":"2022-06-25T20:56:12.610959Z"}}
    cf_matrix = confusion_matrix(y_test, y_pred)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:58:26.795276Z","iopub.execute_input":"2022-06-25T20:58:26.795856Z","iopub.status.idle":"2022-06-25T20:58:26.804841Z","shell.execute_reply.started":"2022-06-25T20:58:26.795806Z","shell.execute_reply":"2022-06-25T20:58:26.803390Z"}}
    accuracy, precision, recall, f1_score = [], [], [], []
    Basis = ['Cosine similarity']

    accuracy.append(scores['test_accuracy'].mean())
    precision.append(scores['test_precision'].mean())
    recall.append(scores['test_recall'].mean())
    f1_score.append(scores['test_f1_score'].mean())

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-25T20:59:00.575649Z","iopub.execute_input":"2022-06-25T20:59:00.576092Z","iopub.status.idle":"2022-06-25T20:59:00.591737Z","shell.execute_reply.started":"2022-06-25T20:59:00.576058Z","shell.execute_reply":"2022-06-25T20:59:00.590990Z"}}
    scoresDict = {'Model': ['Baseline Model'], 'Description': ['MP-Time Model'], 'Basis': Basis,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1Score':f1_score}

    scoresDf = pd.DataFrame(scoresDict)
    return scoresDf