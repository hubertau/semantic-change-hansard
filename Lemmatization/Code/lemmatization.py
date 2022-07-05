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

#from nltk.stem.snowball import SnowballStemmer
import spacy

from multiprocessing import Pool, cpu_count

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print('Hello')

with open('./dfForLemzn.pkl', 'rb') as f:

    partyTimeDf = pickle.load(f)
    
print(partyTimeDf.sort_values(by='LengthLemmas'))
spacyMod = spacy.load('en_core_web_sm')
spacyMod.max_length =104584200
#17417090
#1.37*10000000 


partyTimeDf['Lemmatized']='s'

partyTimeDf.reset_index(inplace=True)

firstHalf = partyTimeDf.iloc[:int(partyTimeDf.shape[0]/2)]
secondHalf = partyTimeDf.iloc[int(partyTimeDf.shape[0]/2):int(partyTimeDf.shape[0])]


labours = firstHalf.iloc[6:8]
#labours
firstLabour = labours.iloc[[0]]

'''def parallelize_dataframe(df, func, num_partitions):
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
        print('Df with shape', df.shape, 'called')
        print('the vocab size is',df['LengthLemmas'])
        df["lemmatized"] = df['sentence'].apply(lambda x:[y.lemma_ for y in spacyMod(x, disable = ['ner','parser'])])
        print('returning',df['df_name'], 'lemmatizing')
        
        dfPath = './' + str(df.iloc[0]['df_name'])+'.pkl'
        print('df path is -',dfPath)
        df.to_pickle(dfPath)
    
        
    elif(df.shape[0]>1):
        print('DF rows more than one, DF is - ', df)
    
    else:
        print('Seems like series gotten. Who are you?',df)
        
        
    return df'''

num_cores = cpu_count()
print('Cores',num_cores)

#conservatives = secondHalf.iloc[[4]]
#print(firstLabour['sentence'])
#print('vocab size',firstLabour['LengthLemmas'])
#print('nowwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',type(firstLabour.at[6,'sentence']))

ind = 19
#lemm = [y.lemma_ for y in spacyMod(partyTimeDf.at[ind,'sentence'], disable = ['ner', 'parser'])]
print('lemm')
partyTimeDf.iloc[ind]['lemmatized']=lemm
print('assigned to the df')

dfPath = './' + str(partyTimeDf.iloc[ind]['df_name'])+'.pkl'
print('df path is -',dfPath)
partyTimeDf.iloc[ind].to_pickle(dfPath)

print('saved')
#print(firstLabour)
#ress = parallelize_dataframe(firstLabour, my_func, num_cores)
print('----------------------------------------------x End of prog x ----------------------------------------')
