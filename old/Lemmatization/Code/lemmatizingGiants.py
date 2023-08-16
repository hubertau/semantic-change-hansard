import pandas as pd
import pickle
import numpy as np

import gensim

from joblib import Parallel, delayed
from scipy import spatial
import functools

#from nltk.stem.snowball import SnowballStemmer
import spacy

from multiprocessing import Pool, cpu_count


spacyMod = spacy.load('en_core_web_sm')
spacyMod.max_length =104584200
#17417090



with open('./dfForLemzn.pkl','rb') as f:
        df = pickle.load(f)
        print('DF is',df)

chunks = 10

df=df[df['df_name']=='df_t2_Conservative']
#print('DF selected is ', df)
print('-------------------------------------------------')


splittable = str(df.at[0,'sentence']).split(' ')
print('The length of the split string in terms of words is',len(splittable))

#splittable = str(df.at[0,'sentence']).split(' ')
#print(splittable)

s=list((*np.array_split(splittable, chunks),))
#print(s)


listOfSplitWords = [ss.tolist() for ss in s]
sentences = [' '.join(w) for w in listOfSplitWords]
party = [df.at[0,'party'] for i in range(chunks)]
df_name  = [df.at[0,'df_name'] for i in range(chunks)]
LengthLemmas=[len(list(ph)) for ph in sentences]
#[df.at[0,'LengthLemmas'] for i in range(chunks)]
Lemmatized = ['s' for i in range(chunks)]
ind = [i for i in range(1,11)]
print('Party and DF name are', party,df_name)


finalDf = pd.DataFrame({'ind':ind, 'party':party, 'df_name':df_name, 'sentence':sentences, 'LengthLemmas':LengthLemmas, 'Lemmatized':Lemmatized})
finalDf=finalDf[finalDf['ind'].isin([1,3,4,5,7])]
#3,7,9,10

print(finalDf)

print('NOW BEGINNING LEMMZN-------------------------------------------------------------------------------------')


def parallelize_dataframe(df, func, num_partitions):
    print(0)
    df_split = np.array_split(df, num_partitions)
    #print(df_split)
    #print(1)
    pool = Pool(num_partitions)
    #print(2)
    temp= pool.map(func, df_split)

    print('Returned to be now concatenated', len(temp))
    df = pd.concat(temp)
    #print('1')
    pool.close()
    pool.join()
    return df


def my_func(df):
    print('Been called')


    if(df.shape[0]==0):
        print('Empty df gotten')

    elif(df.shape[0]==1):
        print('DF BEING PROCESSED --------',df)
        #print('Df',df.at[0,'df_name'],df.at[0,'ind'],' with shape', df.shape, 'being processed')
        print('the vocab size is',df['LengthLemmas'])
        df["lemmatized"] = df['sentence'].apply(lambda x:[y.lemma_ for y in spacyMod(x, disable = ['ner','parser'])])
        print('returning',df['df_name'], 'lemmatizing')

        dfPath = './' + str(df.iloc[0]['df_name'])+str(df.iloc[0]['ind'])+'.pkl'
        print('df path is -',dfPath)
        df.to_pickle(dfPath)


    elif(df.shape[0]>1):
        print('DF rows more than one, DF is - ', df)

    else:
        print('Seems like series gotten. Who are you?',df)

    return df

num_cores = cpu_count()
print('Cores',num_cores)

#conservatives = secondHalf.iloc[[4]]
#print(firstLabour['sentence'])
#print('vocab size',firstLabour['LengthLemmas'])
#print('nowwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',type(firstLabour.at[6,'sentence']))

#ind = 19
#lemm = [y.lemma_ for y in spacyMod(partyTimeDf.at[ind,'sentence'], disable = ['ner', 'parser'])]
#print('lemm')
#partyTimeDf.iloc[ind]['lemmatized']=lemm
#print('assigned to the df')

#dfPath = './' + str(partyTimeDf.iloc[ind]['df_name'])+'.pkl'
#print('df path is -',dfPath)
#partyTimeDf.iloc[ind].to_pickle(dfPath)

#print('saved')
#print(firstLabour)
ress = parallelize_dataframe(finalDf, my_func, num_cores)
print('-----------------ENG----------------')
