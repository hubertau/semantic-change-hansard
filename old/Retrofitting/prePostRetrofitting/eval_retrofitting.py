def retrofitting(factor):
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


    #retrofittingFactor = 'party'
    retrofittingFactor = factor
    if retrofittingFactor =='party':
            retrofittedVectorPath = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\retrofittedVectors.txt"
    elif retrofittingFactor =='partyTime':
        retrofittedVectorPath = r"C:\Users\user\Downloads\semantic-change-hansard-dev\Resources\retrofittedPartyTimeVectors.txt"

    with open(retrofittedVectorPath) as f:

        vecs=[]
        vec=''

        while True:
            line = f.readline()
            if not line: 
                break        
            if(str(list(line)[0]).isalpha()):
                vec=vec.strip()
                if(vec!=''):
                    vecs.append(vec)
                vec = line
            else:
                vec+=line

    vecs = [vec.replace('\n', '')for vec in vecs]
    print(str(len(vecs))+' Retrofitted vectors obtained')

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:17.260690Z","iopub.execute_input":"2022-06-21T16:59:17.261339Z","iopub.status.idle":"2022-06-21T16:59:17.276603Z","shell.execute_reply.started":"2022-06-21T16:59:17.261291Z","shell.execute_reply":"2022-06-21T16:59:17.275567Z"},"jupyter":{"outputs_hidden":false}}
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

    dictKeyVector = {}
    count=0
    for i in range(len(vecs)):

        vec = vecs[i].strip().split(' ')
        # Extracting synonym key
        synKey = vec[0]
        del(vec[0])
        vec=[i for i in vec if i!='']

        if(len(vec)!=300):
            print('Vector with dimension<300', synKey,len(vec))
            count=count+1
        else:
            vec =[float(v) for v in vec]
            dictKeyVector[synKey]=vec
            npVec = np.array(dictKeyVector[synKey])
    print('Count of vectors with fewer dimensions that we will not consider',count)
    dfRetrofitted = pd.DataFrame({'vectorKey':list(dictKeyVector.keys()), 'vectors':list(dictKeyVector.values())})
    dfRetrofitted.head()

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:18.800344Z","iopub.execute_input":"2022-06-21T16:59:18.800810Z","iopub.status.idle":"2022-06-21T16:59:18.808339Z","shell.execute_reply.started":"2022-06-21T16:59:18.800773Z","shell.execute_reply":"2022-06-21T16:59:18.807148Z"},"jupyter":{"outputs_hidden":false}}
    dfRetrofitted.shape

    # %% [markdown]
    # **For party based retrofitting
    # 2071 retrofitted vectors were expected. 
    # 2070 were created. 
    # 55 vectors discarded that had dimensions<300
    # 2015 vectors left** 
    # 
    # 
    # **For party-time retrofitting - From 1962 input vectors, 1634 retrofitted vectors were received(328 were lost during retrofitting, no reason found yet), 
    # Further, 35 vectors have been discarded as the vector dimensions were lost (under 300)
    # Eventually left with 1599**

    # %% [markdown]
    # # **Calculating Cosine similarity**

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:23.601858Z","iopub.execute_input":"2022-06-21T16:59:23.602387Z","iopub.status.idle":"2022-06-21T16:59:23.610292Z","shell.execute_reply.started":"2022-06-21T16:59:23.602355Z","shell.execute_reply":"2022-06-21T16:59:23.609458Z"},"jupyter":{"outputs_hidden":false}}
    # Filtering down words of interest as per those present in our vectors 
    # We're amending the computeAvgVec function accordingly
    # As it calculated based on processing from models, and here we're only taking vectors. Hence this check here too.

    vectorKeys =list(dfRetrofitted['vectorKey'])
    # Extracting words from vectors keys
    words_of_interest = list(set([vk.split('-')[0] for vk in vectorKeys]))
    print(words_of_interest, len(words_of_interest))

    # NOW WE ONLY HAVE THOSE WORDS HERE WHICH ARE PRESENT IN THE VECTORS.

    # %% [markdown]
    # **Functions for cosine similarity computation and to compute the average vector amongst many vectors for a given word**

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:25.482142Z","iopub.execute_input":"2022-06-21T16:59:25.482693Z","iopub.status.idle":"2022-06-21T16:59:25.491724Z","shell.execute_reply.started":"2022-06-21T16:59:25.482661Z","shell.execute_reply":"2022-06-21T16:59:25.490672Z"},"jupyter":{"outputs_hidden":false}}
    # Different from the avg computation function in our other scripts. This works upon vectors instead of models
    def computeAvgVec(mKeys, dicto, w, layerSize=300):
        modelsSum = np.zeros(layerSize)
        for k in mKeys:
            vectorPerModel = dicto[k]
            modelsSum = np.add(modelsSum, vectorPerModel)
        avgEmbedding =np.divide(modelsSum, len(mKeys))
        return avgEmbedding

    def cosine_similarity(vec1, vec2):
      sc = 1-spatial.distance.cosine(vec1, vec2)
      return sc

    cosine_similarity_df = pd.DataFrame(columns = ('Word', 'Cosine_similarity'))

    # %% [markdown]
    # **Compute cosine similarity between avg T1 and T2 vectors**

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:26.992102Z","iopub.execute_input":"2022-06-21T16:59:26.992523Z","iopub.status.idle":"2022-06-21T16:59:27.175849Z","shell.execute_reply.started":"2022-06-21T16:59:26.992490Z","shell.execute_reply":"2022-06-21T16:59:27.174681Z"},"jupyter":{"outputs_hidden":false}}

    t1Keys = [t for t in list(dictKeyVector.keys()) if 't1' in t]
    t2Keys = [t for t in list(dictKeyVector.keys()) if 't2' in t]
    sims= []

    # Compute average of word in T1 and in T2 and store average vectors and cosine difference   
    for word in words_of_interest:

            #Provide a list of keys to average computation model for it to
            #compute average vector amongst these models
            wordT1Keys = [k for k in t1Keys if k.split('-')[0]==word]
            wordT2Keys = [k for k in t2Keys if k.split('-')[0]==word]

            #Since here the key itself contains the word we're not simply sending T1 keys but sending word-wise key
            avgVecT1 = computeAvgVec(wordT1Keys, dictKeyVector, word)
            avgVecT2 = computeAvgVec(wordT2Keys, dictKeyVector, word)

            if(avgVecT1.shape == avgVecT2.shape):
                # Cos similarity between averages
                cosSimilarity = cosine_similarity(avgVecT1, avgVecT2)
                sims.append(cosSimilarity)
            else:
                print('Word not found')
    cosine_similarity_df['Word']=words_of_interest
    cosine_similarity_df['Cosine_similarity']=sims


    #Assigning change and no-change labels as initially expected
    cosine_similarity_df['semanticDifference']=['default' for i in range(cosine_similarity_df.shape[0])]
    cosine_similarity_df.loc[cosine_similarity_df['Word'].isin(change), 'semanticDifference'] = 'change' 
    cosine_similarity_df.loc[cosine_similarity_df['Word'].isin(no_change), 'semanticDifference'] = 'no_change' 
    cosine_similarity_df.sort_values(by='Cosine_similarity').head(10)

    # %% [markdown]
    # # **LOGISTIC REGRESSION**

    # %% [markdown]
    # **Evaluate the performance of retrofitted vectors**

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:28.522125Z","iopub.execute_input":"2022-06-21T16:59:28.522549Z","iopub.status.idle":"2022-06-21T16:59:28.527398Z","shell.execute_reply.started":"2022-06-21T16:59:28.522515Z","shell.execute_reply":"2022-06-21T16:59:28.526162Z"},"jupyter":{"outputs_hidden":false}}
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:29.546787Z","iopub.execute_input":"2022-06-21T16:59:29.547181Z","iopub.status.idle":"2022-06-21T16:59:29.570597Z","shell.execute_reply.started":"2022-06-21T16:59:29.547147Z","shell.execute_reply":"2022-06-21T16:59:29.569344Z"},"jupyter":{"outputs_hidden":false}}
    X = cosine_similarity_df['Cosine_similarity'].values.reshape(-1,1)
    y = cosine_similarity_df['semanticDifference']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

    undersample = RandomUnderSampler(sampling_strategy=1.0)

    X_over, y_over = undersample.fit_resample(X, y)
    X=X_over
    y=y_over

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    logreg = LogisticRegression()
    kf = logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    print('Y value counts',y.value_counts(),'\n')
    print('Y train value counts', y_train.value_counts())

    print(accuracy_score)

    scoring = {'accuracy' : make_scorer(accuracy_score), 
                   'precision' : make_scorer(precision_score,pos_label='change'),
                   'recall' : make_scorer(recall_score,pos_label='change'), 
                   'f1_score' : make_scorer(f1_score,pos_label='change')}

    scores = cross_validate(kf, X, y, cv=10, scoring=scoring,error_score='raise')

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:31.962483Z","iopub.execute_input":"2022-06-21T16:59:31.962891Z","iopub.status.idle":"2022-06-21T16:59:32.219626Z","shell.execute_reply.started":"2022-06-21T16:59:31.962858Z","shell.execute_reply":"2022-06-21T16:59:32.218542Z"},"jupyter":{"outputs_hidden":false}}
    cf_matrix = confusion_matrix(y_test, y_pred)

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:33.514511Z","iopub.execute_input":"2022-06-21T16:59:33.514978Z","iopub.status.idle":"2022-06-21T16:59:33.547014Z","shell.execute_reply.started":"2022-06-21T16:59:33.514942Z","shell.execute_reply":"2022-06-21T16:59:33.545757Z"},"jupyter":{"outputs_hidden":false}}

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:35.527957Z","iopub.execute_input":"2022-06-21T16:59:35.528435Z","iopub.status.idle":"2022-06-21T16:59:35.534346Z","shell.execute_reply.started":"2022-06-21T16:59:35.528398Z","shell.execute_reply":"2022-06-21T16:59:35.533097Z"},"jupyter":{"outputs_hidden":false}}
    accuracy, precision, recall, f1_score = [], [], [], []
    retrofittingBasis = ['Vectors retrofitted based on '+retrofittingFactor]

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:39.304672Z","iopub.execute_input":"2022-06-21T16:59:39.305080Z","iopub.status.idle":"2022-06-21T16:59:39.312109Z","shell.execute_reply.started":"2022-06-21T16:59:39.305048Z","shell.execute_reply":"2022-06-21T16:59:39.310724Z"},"jupyter":{"outputs_hidden":false}}
    #scoresDf = pd.DataFrame(columns= ['retrofittingBasis','Accuracy','Precision','Recall','F1Score'])
    #scoresDf.append(['party', scores['test_accuracy'].mean(), scores['test_precision'].mean(),scores['test_recall'].mean(), scores['test_f1_score'].mean()],axis=1)
    accuracy.append(scores['test_accuracy'].mean())
    precision.append(scores['test_precision'].mean())
    recall.append(scores['test_recall'].mean())
    f1_score.append(scores['test_f1_score'].mean())

    # %% [code] {"execution":{"iopub.status.busy":"2022-06-21T16:59:50.923409Z","iopub.execute_input":"2022-06-21T16:59:50.923926Z","iopub.status.idle":"2022-06-21T16:59:50.943043Z","shell.execute_reply.started":"2022-06-21T16:59:50.923888Z","shell.execute_reply":"2022-06-21T16:59:50.941795Z"},"jupyter":{"outputs_hidden":false}}
    
    scoresDict = {'Model': ['Retrofitted Model'], 'Description': retrofittingBasis, 'Basis': ['Cosine Similarity'],'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1Score':f1_score}
 

    scoresDf = pd.DataFrame(scoresDict)
    return scoresDf