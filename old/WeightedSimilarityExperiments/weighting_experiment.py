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

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

import scipy.stats as stats
from sklearn.metrics import confusion_matrix

import imblearn
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

import seaborn as sns

import os
# Load 24 Word2Vec models of MPs in T1 & T2

def weighting():
    dictOfModels = {}

    folderPath = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\24-aligned-models-by-mp-and-time"

    #folderPath = '/kaggle/input/aligned24mptimemodels/kaggle/working/24-aligned-models-by-mp-and-time'


    for file in os.listdir(folderPath):
        filePath = folderPath + '/' + file

        #To accommodate errors while picking up corresponding numpy files of gensim models
        if(len(filePath.split('.'))>1):
            continue
        else:
            model = gensim.models.Word2Vec.load(filePath)
            dictOfModels[file] = model

    #dictOfModels

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T07:19:57.016486Z","iopub.execute_input":"2022-06-22T07:19:57.016881Z","iopub.status.idle":"2022-06-22T07:19:57.132882Z","shell.execute_reply.started":"2022-06-22T07:19:57.016849Z","shell.execute_reply":"2022-06-22T07:19:57.131661Z"},"jupyter":{"outputs_hidden":false}}
    # Extract time, party, MP ID information from name and store in Dataframes
    brexitEmbeddings = {}

    for k in dictOfModels.keys():

        time = k.split('df_')[1].split('_')[0]
        mpId = k.split('df_')[1].split('_')[1]
        party = k.split('df_')[1].split('_')[2]

        brexEmbDf =pd.DataFrame()
        brexEmbDf['model'] = [dictOfModels[k]]
        brexEmbDf['modelKey'] = k
        brexEmbDf['time'] = time
        brexEmbDf['mpId'] = mpId
        brexEmbDf['party'] = party

        brexitEmbeddings[k] = brexEmbDf


    brexitDf = pd.DataFrame()

    for v in brexitEmbeddings.values():
        brexitDf = brexitDf.append(v)

    brexitDf.reset_index(inplace=True)

    def cosine_similarity(vec1, vec2):
      sc = 1-spatial.distance.cosine(vec1, vec2)
      return sc

    # Modified Similarity working upon a DF of records comprising 2 vectors (2 MP-time records in a row)
    def calc_similarity(df,x):
        #sim(U,V) = ( x )*cosine(U,V) + (1-x)*same_party(U,V)
        df['similarity']=df['cossims']*x + (1-x)*(df['sameParty'])
        return df
    
    def calculate_similarities(change):

        simFactor = 0.8
        words = []
        cosineSimilarities = []
        medianSimilarities = []
        meanSimilarities = []
        stdSimilarities = []

        for word in change:

            count =0
            keys=list(brexitEmbeddings.keys())


            if word in brexitEmbeddings[keys[0]]['model'][0].wv.index_to_key:

                simMat = pd.DataFrame()

                parties1 = []
                times1 = []
                mps1 = []
                parties2 = []
                times2 = []
                mps2 = []
                cossims= []

                for i, k in enumerate(keys):

                        constKey = brexitEmbeddings[k]
                        constModel = brexitEmbeddings[k]['model'][0]
                        constParty = brexitEmbeddings[k]['party'][0]
                        constTime = brexitEmbeddings[k]['time'][0]
                        constMP = brexitEmbeddings[k]['mpId'][0]


                        for j in range(0,len(keys)):

                            count = count+1
                            nextVec=brexitEmbeddings[keys[j]]
                            nextVecModel = nextVec['model'][0]
                            nextVecParty = nextVec['party'][0]
                            nextVecTime = nextVec['time'][0]
                            nextVecMP = nextVec['mpId'][0]

                            cossim = cosine_similarity(constModel.wv[word], nextVecModel.wv[word])

                            parties1.append(constParty)
                            parties2.append(nextVecParty)
                            times1.append(constTime)
                            times2.append(nextVecTime)
                            mps1.append(constMP)
                            mps2.append(nextVecMP)
                            cossims.append(cossim)

                simMat['parties1']=parties1
                simMat['times1'] = times1
                simMat['mps1'] = mps1
                simMat['parties2']=parties2
                simMat['times2'] = times2
                simMat['mps2'] = mps2
                simMat['cossims'] = cossims


                #Adding Same party factor as a column
                simMat['sameParty']=0

                simMat.loc[simMat['parties1']==simMat['parties2'],'sameParty']=1

                #print('MP-time combinations DFs shape is ',simMat.shape)
                print(simMat)
                # Calculating Modified similarity
                similarityDF = calc_similarity(simMat,simFactor)

                medianSimilarity = similarityDF['similarity'].median()
                medianCosineSimilarity = similarityDF['cossims'].median()

                meanSimilarity = similarityDF['similarity'].mean()
                stdSimilarity = similarityDF['similarity'].std()


                #print('The median-similarity for ',word, 'is ', medianSimilarity)

                words.append(word)
                medianSimilarities.append(medianSimilarity)
                cosineSimilarities.append(medianCosineSimilarity)
                meanSimilarities.append(meanSimilarity)
                stdSimilarities.append(stdSimilarity)


            else:
                print('Word', word, 'not found in the vocab of models')

        medianSimilarityDf = pd.DataFrame({'Word':words, 'median_similarity':medianSimilarities, 'median_cosineSimilarity':cosineSimilarities, 'meanSimilarity':meanSimilarities, 'stdSimilarity':stdSimilarities})
        return medianSimilarityDf

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T07:19:57.331459Z","iopub.execute_input":"2022-06-22T07:19:57.332297Z","iopub.status.idle":"2022-06-22T07:19:57.349310Z","shell.execute_reply.started":"2022-06-22T07:19:57.332244Z","shell.execute_reply":"2022-06-22T07:19:57.348042Z"},"jupyter":{"outputs_hidden":false}}

    def calculate_similarities(change,antiPartyFactor):

        words = []
        cosineSimilarities = []
        medianSimilarities = []
        meanSimilarities = []
        stdSimilarities = []

        keys = list(brexitEmbeddings.keys())

        keyst1=[kt1 for kt1 in keys if 't1' in kt1]
        keyst2=[kt2 for kt2 in keys if 't2' in kt2]
        #print(len(keyst1), len(keyst2))


        for word in change:

            count =0

            if word in brexitEmbeddings[keys[0]]['model'][0].wv.index_to_key:

                simMat = pd.DataFrame()

                parties1 = []
                times1 = []
                mps1 = []
                parties2 = []
                times2 = []
                mps2 = []
                cossims= []

                for i, k in enumerate(keyst1):

                        constKey = brexitEmbeddings[k]
                        constModel = brexitEmbeddings[k]['model'][0]
                        constParty = brexitEmbeddings[k]['party'][0]
                        constTime = brexitEmbeddings[k]['time'][0]
                        constMP = brexitEmbeddings[k]['mpId'][0]


                        for j in range(0,len(keyst2)):

                            count = count+1
                            nextVec=brexitEmbeddings[keyst2[j]]
                            nextVecModel = nextVec['model'][0]
                            nextVecParty = nextVec['party'][0]
                            nextVecTime = nextVec['time'][0]
                            nextVecMP = nextVec['mpId'][0]

                            cossim = cosine_similarity(constModel.wv[word], nextVecModel.wv[word])

                            parties1.append(constParty)
                            parties2.append(nextVecParty)
                            times1.append(constTime)
                            times2.append(nextVecTime)
                            mps1.append(constMP)
                            mps2.append(nextVecMP)
                            cossims.append(cossim)

                simMat['parties1']=parties1
                simMat['times1'] = times1
                simMat['mps1'] = mps1
                simMat['parties2']=parties2
                simMat['times2'] = times2
                simMat['mps2'] = mps2
                simMat['cossims'] = cossims


                #Adding Same party factor as a column
                simMat['sameParty']=0

                simMat.loc[simMat['parties1']==simMat['parties2'],'sameParty']=1

                #print('MP-time combinations DFs shape is ',simMat.shape)
                #print(simMat.shape)
                #print(simMat)

                # Calculating Modified similarity
                similarityDF = calc_similarity(simMat,antiPartyFactor)

                medianSimilarity = similarityDF['similarity'].median()
                medianCosineSimilarity = similarityDF['cossims'].median()

                meanSimilarity = similarityDF['similarity'].mean()
                stdSimilarity = similarityDF['similarity'].std()


                #print('The median-similarity for ',word, 'is ', medianSimilarity)

                words.append(word)
                medianSimilarities.append(medianSimilarity)
                cosineSimilarities.append(medianCosineSimilarity)
                meanSimilarities.append(meanSimilarity)
                stdSimilarities.append(stdSimilarity)


            else:
                print('Word', word, 'not found in the vocab of models')

        medianSimilarityDf = pd.DataFrame({'Word':words, 'median_similarity':medianSimilarities, 'median_cosineSimilarity':cosineSimilarities, 'meanSimilarity':meanSimilarities, 'stdSimilarity':stdSimilarities})
        return medianSimilarityDf

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T07:19:57.494218Z","iopub.execute_input":"2022-06-22T07:19:57.494968Z","iopub.status.idle":"2022-06-22T07:19:57.510364Z","shell.execute_reply.started":"2022-06-22T07:19:57.494932Z","shell.execute_reply":"2022-06-22T07:19:57.509362Z"},"jupyter":{"outputs_hidden":false}}
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



    keys=list(brexitEmbeddings.keys())
    wo_interest = change+no_change

    #Filtering for words present in models' vocab
    wo_interest = list(set(wo_interest).intersection(brexitEmbeddings[keys[0]]['model'][0].wv.index_to_key))

    def weighting_experiment(x):
        similarity = calculate_similarities(wo_interest,antiPartyFactor=x)

        similarity.loc[similarity['Word'].isin(change), 'change']='change'
        similarity.loc[similarity['Word'].isin(no_change), 'change']='no_change'

        words_of_interest = similarity[similarity['Word'].isin(change+no_change)]

        words_of_interest.loc[words_of_interest['Word'].isin(change), 'semanticDifference'] = 'change' 
        words_of_interest.loc[words_of_interest['Word'].isin(no_change), 'semanticDifference'] = 'no_change'

        #X = words_of_interest['median_similarity'].values.reshape(-1,1)
        X = words_of_interest[['meanSimilarity','stdSimilarity']]
        y = words_of_interest['semanticDifference']
        undersample = RandomUnderSampler(sampling_strategy=1.0)

        X_over, y_over = undersample.fit_resample(X, y)
        X=X_over
        y=y_over

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=152)

        logreg = LogisticRegression()
        kf = logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        scoring = {'accuracy' : make_scorer(accuracy_score), 
                   'precision' : make_scorer(precision_score,pos_label='change'),
                   'recall' : make_scorer(recall_score,pos_label='change'), 
                   'f1_score' : make_scorer(f1_score,pos_label='change')}

        #scores = cross_val_score(kf, X, y, cv=10)
        scores = cross_validate(kf, X, y, cv=10, scoring=scoring,error_score='raise')

        return ({'Accuracy': scores['test_accuracy'].mean(), 'Precision': scores['test_precision'].mean(), 'Recall': scores['test_recall'].mean(), 'F1Score': scores['test_f1_score'].mean()})

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-22T07:21:17.804325Z","iopub.execute_input":"2022-06-22T07:21:17.804723Z","iopub.status.idle":"2022-06-22T07:21:35.495865Z","shell.execute_reply.started":"2022-06-22T07:21:17.804692Z","shell.execute_reply":"2022-06-22T07:21:35.494663Z"},"jupyter":{"outputs_hidden":false}}
    resultsDF = pd.DataFrame()
    antiPartyFactors = [0, 0.25,0.5,0.75,1] 
    for x in antiPartyFactors:
        resultMetrics = weighting_experiment(x)
        resultMetrics = pd.DataFrame([resultMetrics])
        resultsDF = pd.concat([resultsDF,resultMetrics],ignore_index=True)
    
    #Model and Basis needed additionally
    antiPartyFactors = ['Weighted model with x as '+str(x) for x in antiPartyFactors]
    resultsDF.insert(0,'Model',antiPartyFactors)
    resultsDF.insert(1, 'Description', 'Corpus split by same MP-time basis')
    resultsDF.insert(2, 'Basis', ['Modified cos similarity inc. same-party factor' for i in range(len(antiPartyFactors))])
    return resultsDF   
    
