"""
Given input data, lemmatize and get metadata
"""

import ast
import csv
import datetime
import functools
import multiprocessing
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from csv import reader

import click
from dateutil.relativedelta import relativedelta
import gensim
import h5py
import imblearn
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import researchpy as rp
import spacy
from genericpath import isfile
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import retrofit
from scipy import spatial
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import cross_validate, train_test_split
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from tqdm import tqdm


class ParliamentDataHandler(object):

    def __init__(self, data, tokenized, data_filename = None, verbosity=0):
        self.unsplit_data = data
        self.tokenized = tokenized
        self.split_complete = False
        self.data_filename = data_filename
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_csv(cls, unsplit_data, tokenized=False):
        df = pd.read_csv(unsplit_data)
        return cls(df, tokenized=tokenized, data_filename=unsplit_data)

    def _tokenize_one_sentence(self, sentence):
        sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        return tokens

    def tokenize_data(self, tokenized_data_dir=None, overwrite = False):
        if self.data_filename and tokenized_data_dir:
            original_filename = os.path.split(self.data_filename)[-1]
            original_filename = original_filename.split('.')[0]
            savepath = os.path.join(
                tokenized_data_dir,
                f'Tokenized_{original_filename}.pkl')
        else:
            savepath = None
        if self.tokenized:
            return None
        elif (os.path.isfile(savepath) and overwrite) or not os.path.isfile(savepath):
            tokens = self.unsplit_data.text.apply(self._tokenize_one_sentence)
            self.unsplit_data['tokenized'] = tokens
            self.unsplit_data.to_pickle(savepath)
            self.logger.info(f'Saved to {savepath}')
        elif (os.path.isfile(savepath) and not overwrite):
            self.logger.info(f'Loading in tokenized data from {savepath}')
            self.unsplit_data = pd.read_pickle(savepath)

    # def remove_stopwords_from_tokenized(self):
    #     #TODO: FIX THIS FUNCTION
    #     self.unsplit_data.loc[:,'tokenized'] = self.unsplit_data['tokenized'].apply(len)

    def split_by_date(self, date, split_range):
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if split_range is not None:
            leftbound  = date - relativedelta(years=split_range)
            rightbound = date + relativedelta(years=split_range)
        else:
            leftbound  = date - relativedelta(years=100)
            rightbound = date + relativedelta(years=100)

        self.unsplit_data['datetime'] = pd.to_datetime(self.unsplit_data['date'])

        self.data_t1 = self.unsplit_data[(self.unsplit_data['datetime'] > leftbound) & (self.unsplit_data['datetime'] <= date)]
        self.data_t2 = self.unsplit_data[(self.unsplit_data['datetime'] > date) & (self.unsplit_data['datetime'] < rightbound)]
        self.split_complete = True

    # def obtain_unique(self, df, by='party'):
    #     if by == 'party':
    #         pass
    #     elif by == 'mp':
    #         by = 'speaker'
    #     unique = list(df[by].unique())
    #     return unique

    def preprocess(self, model = None, model_output_dir = None, retrofit_outdir=None, overwrite=None):
        """TODO: Use this function to unify the retrofit prep, the tokenising, splitting of speeches, etc. so this is not duplicated in subsequent processes"""
        assert model in ['retrofit', 'retro', 'whole', 'speaker']
        self.model_type = model
        if self.model_type in ['retrofit', 'retro']:
            self.logger.info(f'PREPROCESS: Running preprocessing for retrofit.')
            self.retrofit_prep(retrofit_outdir=retrofit_outdir, overwrite = overwrite)
        elif self.model_type == 'speaker':
            self.logger.info(f'PREPROCESS - SPEAKER - None required.')
        elif self.model_type == 'whole':
            self.logger.info(f'PREPROCESS - WHOLE - None required.')

    def process_speaker(self, model_output_dir, min_vocab_size=10000, overwrite=False):
        """Function to load in speaker models, filter by vocab size if necessary.
        """
        # collect keys of models that are over the threshold
        self.valid_split_speeches_by_mp = []
        first = True
        self.dictOfModels = {
            't1': [],
            't2': []
        }
        total_words = set()
        for model_savepath in self.speaker_saved_models:
            model = gensim.models.Word2Vec.load(model_savepath)
            # if len(model.wv.index_to_key) < min_vocab_size:
                # continue
            if 't1' in model_savepath:
                self.dictOfModels['t1'].append(model)
            else:
                self.dictOfModels['t2'].append(model)
            # else:
            self.valid_split_speeches_by_mp.append(model_savepath)
            if first:
                total_words = set(model.wv.index_to_key)
                # print(total_words)
                first = False
            else:
                total_words.update(model.wv.index_to_key)
        self.logger.info(f'PREPROCESS - SPEAKER - Total words: {len(total_words)}')

        # print, for informational purposes, the number of models that are over the threshold
        self.logger.info(f'PREPROCESS - SPEAKER - Number of valid models: {len(self.valid_split_speeches_by_mp)}')

        avg_vec_savepath_t1 = os.path.join(model_output_dir,'average_vecs_t1.bin')
        avg_vec_savepath_t2 = os.path.join(model_output_dir,'average_vecs_t2.bin')

        if ((os.path.isfile(avg_vec_savepath_t1) or os.path.isfile(avg_vec_savepath_t2)) and overwrite) or not (os.path.isfile(avg_vec_savepath_t1) and os.path.isfile(avg_vec_savepath_t2)):
            average_vecs = {
                't1': {},
                't2': {}
            }
            self.cosine_similarity_df = pd.DataFrame(columns = ('Word', 'Cosine_similarity'))
            self.logger.info(f'PREPROCESS - SPEAKER - CALCULATING AVERAGE VECTORS')
            for word in tqdm(total_words):
                if self.verbosity > 0:
                    self.logger.info(f'PREPROCESS - SPEAKER - getting average vector for {word}')
                avgVecT1 = self.computeAvgVec(word, time='t1')
                avgVecT2 = self.computeAvgVec(word, time='t2')

                if(np.sum(avgVecT1)==0 or np.sum(avgVecT2)==0):
                    if self.verbosity > 0:
                        print(str(word) + ' Word not found')
                    continue
                else:
                    # build results of average vec to save.
                    average_vecs['t1'][word] = avgVecT1
                    average_vecs['t2'][word] = avgVecT2

                    # Cos similarity between averages
                    cosSimilarity = self.cosine_similarity(avgVecT1, avgVecT2)
                    insert_row = {
                        "Word": word,
                        "Cosine_similarity": cosSimilarity
                    }

                    self.cosine_similarity_df = pd.concat([self.cosine_similarity_df, pd.DataFrame([insert_row])])

            self._save_word2vec_format(
                fname = avg_vec_savepath_t1,
                vocab = average_vecs['t1'],
                vector_size = average_vecs['t1'][list(average_vecs['t1'].keys())[0]].shape[0]
            )
            self.logger.info(f'PREPROCESS - SPEAKER - Average vectors for t1 saved to {avg_vec_savepath_t1}')
            self._save_word2vec_format(
                fname = avg_vec_savepath_t2,
                vocab = average_vecs['t2'],
                vector_size = average_vecs['t2'][list(average_vecs['t2'].keys())[0]].shape[0]
            )
            self.logger.info(f'PREPROCESS - SPEAKER - Average vectors for t2 saved to {avg_vec_savepath_t2}')

            self.model1 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t1, binary=True)
            self.model2 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t2, binary=True)

        else:
            self.model1 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t1, binary=True)
            self.logger.info(f'PREPROCESS - SPEAKER - Average vectors for t1 loaded in from {avg_vec_savepath_t1}')
            self.model2 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t2, binary=True)
            self.logger.info(f'PREPROCESS - SPEAKER - Average vectors for t2 loaded in from {avg_vec_savepath_t2}')

            self.cosine_similarity_df = pd.DataFrame(columns = ('Word', 'Cosine_similarity'))
            for word in self.model1.index_to_key:
                avgVecT1 = self.model1[word]
                avgVecT2 = self.model2[word]

                cosSimilarity = self.cosine_similarity(avgVecT1, avgVecT2)
                self.cosine_similarity_df = self.cosine_similarity_df.append(
                    {'Word': word, 'Cosine_similarity': cosSimilarity},
                    ignore_index=True
                )

    def _intersection_align_gensim(self, m1, m2, words=None):
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
        # print(1)
        # Get the vocab for each model
        vocab_m1 = set(m1.wv.index_to_key)
        vocab_m2 = set(m2.wv.index_to_key)
        # print(2)

        # Find the common vocabulary
        common_vocab = vocab_m1 & vocab_m2
        if words: common_vocab &= set(words)
        # print(3)

        # If no alignment necessary because vocab is identical...
        if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
            return (m1,m2)

        # Otherwise sort by frequency (summed for both)
        common_vocab = list(common_vocab)
        common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
        # print(len(common_vocab))

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

            # print(len(m.wv.key_to_index), len(m.wv.vectors))
            # if(len(m.wv.key_to_index)==135):
                # print('Common vocab is', common_vocab)

        return (m1,m2)

    # Function to align two spaces with orthogunal procrustes
    def _smart_procrustes_align_gensim(self, base_embed, other_embed, words=None):
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
        # print(4)

        # make sure vocabulary and indices are aligned
        in_base_embed, in_other_embed = self._intersection_align_gensim(base_embed, other_embed, words=words)

        in_base_embed.wv.fill_norms(force=True)
        in_other_embed.wv.fill_norms(force=True)

        # print(5)

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

        return other_embed

    def split_speeches(self, by='party'):
        """Function to split loaded data by party

        Returns:
            dictSpeechesbyParty (dict): dict of partty:
        """


        if by == 'party':
            pass
        elif by == 'mp':
            by = 'speaker'
        split_t1 = list(self.data_t1[by].unique())
        split_t2 = list(self.data_t2[by].unique())
        total_split = set(split_t1+split_t2)

        splitspeeches = {}

        for p in total_split:

            for dfTime in ['df_t1','df_t2']:

                tempDf = pd.DataFrame()
                tempList = []
                dfName = f"{dfTime}_{p}"

                if(dfTime == 'df_t1'):
                    tempDf = self.data_t1[self.data_t1[by]==p].copy()

                elif (dfTime == 'df_t2'):
                    tempDf = self.data_t2[self.data_t2[by]==p].copy()

                if (tempDf.shape[0]==0):
                    continue

                # used to index 'Lemmas'
                tempDf.loc[:,'Lemmas'] = tempDf['tokenized']
                tempList.extend(tempDf['Lemmas'].to_list())
                split = tempDf[by].iat[0]
                party = tempDf['party'].iat[0]

                #Flatten the list so it's not a list of lists
                # 2022-10-10: This is the offending line that splits words into letters screwing up the subsequent stuff.
                tempList = [item for sublist in tempList for item in sublist]

                tempDf = pd.DataFrame([[split, tempList, party]],columns=[by, 'Lemmas', 'party'])
                splitspeeches[dfName]= tempDf
                splitspeeches[dfName]['df_name'] = dfName

        return splitspeeches

    def split_speeches_df(self, by='party'):
        """Function to split loaded data by party

        Returns:
            dictSpeechesbyParty (dict): dict of partty:
        """


        if by == 'party':
            splitspeeches = pd.DataFrame(columns=['df_name' ,'Lemmas', 'party'])
        elif by == 'mp' or by == 'speaker':
            by = 'speaker'
            splitspeeches = pd.DataFrame(columns=['df_name', by ,'Lemmas', 'party'])
        split_t1 = list(self.data_t1[by].unique())
        split_t2 = list(self.data_t2[by].unique())
        total_split = set(split_t1+split_t2)


        for p in total_split:

            for dfTime in ['df_t1','df_t2']:

                tempDf = pd.DataFrame()
                tempList = []
                dfName = f"{dfTime}_{p}"

                if(dfTime == 'df_t1'):
                    tempDf = self.data_t1[self.data_t1[by]==p].copy()

                elif (dfTime == 'df_t2'):
                    tempDf = self.data_t2[self.data_t2[by]==p].copy()

                if (tempDf.shape[0]==0):
                    continue

                # used to index 'Lemmas'
                tempDf.loc[:,'Lemmas'] = tempDf['tokenized']
                tempList.extend(tempDf['Lemmas'].to_list())
                split = tempDf[by].iat[0]
                party = tempDf['party'].iat[0]

                #Flatten the list so it's not a list of lists
                # 2022-10-10: This is the offending line that splits words into letters screwing up the subsequent stuff.
                tempList = [item for sublist in tempList for item in sublist]

                tempDf = pd.DataFrame([[split, tempList, party]],columns=[by, 'Lemmas', 'party'])
                tempDf['df_name'] = dfName
                splitspeeches = pd.concat([splitspeeches, tempDf], axis=0)
                # splitspeeches[dfName]= tempDf
                # splitspeeches[dfName]['df_name'] = dfName

        return splitspeeches

    def stem_and_lemmatize(self, change, outdir):

        dictSpeechesByParty = self.split_speeches()
        print('SPLIT BY PARTY COMPLETE')

        partyTimeDf = pd.DataFrame(columns = ['party', 'Lemmas', 'df_name'])
        for val in list(dictSpeechesByParty.values()):
            partyTimeDf = partyTimeDf.append(val)

        partyTimeDf['LengthLemmas'] = partyTimeDf.Lemmas.map(len)
        partyTimeDf.agg(Max=('LengthLemmas', max), Min=('LengthLemmas', 'min'), Mean=('LengthLemmas', np.mean))

        partyTimeDf = partyTimeDf.reset_index()
        partyTimeDf['lemmas_delist'] = [','.join(map(str, l)) for l in partyTimeDf['Lemmas']]

        # get overlapping words
        overlappingWords = []
        for ind in partyTimeDf.index:
            partyVocabInTime = partyTimeDf.at[ind, 'Lemmas']

            overlap = list(set(partyVocabInTime).intersection(change))
            overlappingWords.append(overlap)

        partyTimeDf['overlapping_words'] = overlappingWords
        partyTimeDf['overlapCount'] = partyTimeDf.overlapping_words.map(len)

        partyTimeDf['overlap2_delist'] = [','.join(map(str, l)) for l in partyTimeDf['overlapping_words']]

        # wtf is this?!
        partyTimeDfOverridden = partyTimeDf[partyTimeDf['overlap2_delist'].str.contains('leave')]
        for term in ['brexit','remain','cliff','exit','trigger','triggered','triggering','withdraw','remainers','bill']:
            partyTimeDfOverridden = partyTimeDfOverridden[partyTimeDfOverridden['overlap2_delist'].str.contains(term)]

        partyTimeDfOverridden['stemmed']=['s' for i in range(len(partyTimeDfOverridden))]
        partyTimeDfOverridden=partyTimeDfOverridden.reset_index()

        stemmer = SnowballStemmer("english")
        colIndex = partyTimeDfOverridden.columns.get_loc('stemmed')
        print(f"{len(partyTimeDfOverridden)} rows to process")
        for index, row in partyTimeDfOverridden.iterrows():
            print(index,row['party'])
            # print(row['Lemmas'])
            # if pd.isnull(row['Lemmas']):
                # print('no lemmas, continuing...')
                # continue
            if(len(partyTimeDfOverridden.iat[index,colIndex])>1):
                print(partyTimeDfOverridden.iat[index,colIndex][0])
                print('Already set')
                continue
            else:
            #=='s' or partyTimeDfOverridden.iat[index,colIndex].isnull()):
                print('Stemming')
                stemmed=  [stemmer.stem(y) for y in partyTimeDfOverridden.at[index,'Lemmas']] # Stem every word.
                print(len(stemmed))
                #partyTimeDfOverridden.loc[index, 'stemmed'] = stemmed
                partyTimeDfOverridden.iat[index,colIndex] = stemmed

        print('Stemming complete')
        listDfsKeep = partyTimeDfOverridden['df_name'].to_list()
        # Dropping key-value pairs from dictionary where the key doesn't match

        for k,v in list(dictSpeechesByParty.items()):
            if (k not in listDfsKeep):
                del dictSpeechesByParty[k]

        len(dictSpeechesByParty.values())

        lemList = list(partyTimeDfOverridden['stemmed'])
        #print(lemList[0][0])

        for ind,k in enumerate(dictSpeechesByParty.keys()):
            print(ind,k)
            dictSpeechesByParty[k].at[0,'stemmed']='s'
            # print(dictSpeechesByParty['df_t2_Labour (Co-op)'].dtypes)
            stInd = dictSpeechesByParty[k].columns.get_loc('stemmed')
            dictSpeechesByParty[k].iat[0,stInd]=lemList[ind]

        dictOfModels = {}
        count = 1

        print('GENERATE WORDTOVEC')
        for dframe in dictSpeechesByParty: 

            # Doing in batches since notebook RAM crashe
            print(dictSpeechesByParty[dframe]['df_name'])
            print('Hello', dictSpeechesByParty[dframe]['stemmed'])
            model = gensim.models.Word2Vec(
                dictSpeechesByParty[dframe]['stemmed'],
                min_count=1,
                vector_size=300,
                window = 5,
                sg = 1
            )

            # Also saving model in a dict and exporting

            modelName ='model_'+ dframe
            print('model number', count, modelName)

            dictOfModels[dframe] = model
            #model.save(os.path.join(models_folder, modelName))
            count = count +1

        modelsToAlign = list(dictOfModels.values())
        for i in range(0,len(modelsToAlign)-1):
            functools.reduce(self._smart_procrustes_align_gensim, modelsToAlign)

        for ind in range(0,len(listDfsKeep)-1):
            if(len(dictOfModels[listDfsKeep[ind]].wv.index_to_key)!=len(dictOfModels[listDfsKeep[ind+1]].wv.index_to_key)):
                print('Vocabs not similar')

        print('Vocab Size', len(dictOfModels[listDfsKeep[ind]].wv.index_to_key))

        print('SAVING MODELS')
        os.makedirs(outdir, exists_ok=True)
        for k in dictOfModels.keys(): 
            dictOfModels[k].save(os.path.join(outdir, k))

    def retrofit_prep(self, retrofit_outdir=None, overwrite=False):
        if retrofit_outdir:
            assert os.path.isdir(retrofit_outdir)
            self.retrofit_outdir = retrofit_outdir
        if 'parliament' in self.unsplit_data.columns:
            self.parliament_name = self.unsplit_data['parliament'].iat[0]
        else:
            self.parliament_name = 'UNKNOWN PARLIAMENT'

        retrofit_prep_savepath = os.path.join(retrofit_outdir, f'retrofit_prep.json')
        self.logger.info(f'Retrofit Prep Path is {retrofit_prep_savepath}')

        self.logger.info('Running retrofit prep')
        if os.path.isfile(retrofit_prep_savepath) and overwrite or not os.path.isfile(retrofit_prep_savepath):

            self.logger.info('Splitting speeches')

            total_mpTimedf = self.split_speeches_df(by='speaker')
            total_mpTimedf['lemmas_delist'] = [','.join(map(str, l)) for l in total_mpTimedf['Lemmas']]

            total_mpTimedf['LengthLemmas'] = total_mpTimedf.Lemmas.map(len)

            total_mpTimedf.to_json(retrofit_prep_savepath, orient='split', index=False)

            self.retrofit_prep_df = total_mpTimedf

            self.logger.info(f'Retrofit prep saved to {retrofit_prep_savepath}')

        else:

            self.logger.info('Loading in prep from before...')

            self.retrofit_prep_df = pd.read_json(retrofit_prep_savepath, orient='split')
            self.logger.info(f'Retrofit prep loaded in from {retrofit_prep_savepath}, key = {self.parliament_name}')

    def retrofit_create_synonyms_party(self, data, word, factor):
        parties = list(data.party.value_counts().index)
        dictOfSynonyms={}

        # Iterate parties & create synonyms where more than one record for a party
        for p in parties:

            partySynonyms=[]
            partyDf = data[data['party']==p]
            speaker_ids=list(partyDf['speaker'].unique())

            times=list(partyDf['df_name'])
            times = [t.split('_')[1] for t in times]

            # To fix party names like 'Scottish National Party by inserting hyphens between
            if(len(p.split(' '))>1):
                splat = p.split(' ')
                p = '-'.join(splat)

            for ind, name in enumerate(speaker_ids):

                # Concatenating speaker first and last names with '-'    
                name = name.replace(' ','-')

                #Creating synonym string or key 
                syn_str = f"{word}-{times[ind]}-{name}-{p}"
                partySynonyms.append(syn_str)

            dictOfSynonyms[p]=partySynonyms
        #Making pairs
        synonyms=[]
        for k in dictOfSynonyms.keys():
            word_mps_party = dictOfSynonyms[k]
            # Proceed to make pairs only if more than one record per party
            if(len(word_mps_party)>1):
                for i,rec in enumerate(word_mps_party):
                    for j in range(i+1,len(word_mps_party)):
                        # --------------- IF MAKING PAIRS ON PARTY-TIME BASIS, THIS CODE IS THE DIFFERENTIATING BIT---
                        if(factor=='party-time'):
                            if(rec.split('-')[1]==word_mps_party[j].split('-')[1]):
                                syntup = (rec,word_mps_party[j])
                                synonyms.append(syntup)
                        else:
                            syntup = (rec,word_mps_party[j])
                            synonyms.append(syntup)
        return synonyms

    def retrofit_create_synonyms(self, data, word, factor):

        # For both party and party-time basis
        if(factor=='party' or factor=='party-time'):
            synonyms = self.retrofit_create_synonyms_party(data, word, factor)
        return synonyms

    def retrofit_main_create_synonyms(self, overwrite=False):

        assert self.retrofit_prep_df is not None

        synonymFactor = 'party'
        self.synPicklePath = os.path.join(self.retrofit_outdir, f'synonymsParty_{self.parliament_name}.pkl')
        self.synTextPath = os.path.join(self.retrofit_outdir, f'synonymsParty_{self.parliament_name}.txt')

        self.logger.info(f'Retrofit: Processing Synonyms...')
        if ((os.path.isfile(self.synPicklePath) and os.path.isfile(self.synTextPath)) and overwrite) or not (os.path.isfile(self.synPicklePath) and os.path.isfile(self.synTextPath)):

            allSynonyms=[]
            for word in self.words_of_interest:
                synonymsPerWord = self.retrofit_create_synonyms(self.retrofit_prep_df,word,synonymFactor)
                #print(len(synonyms)) #Verify length of synonyms
                allSynonyms.append(synonymsPerWord)
            #Here it is 84 , which is sum of combinations made 
            #for the three parties (13,3,3)=> no. of combinations is (78,3,3), 78+3+3= 84, hence verified. 

            brexitSynonyms = allSynonyms

            # We're capturing synonyms of all words of interest regardless of whether they're part of the models' vocab
            # Since the same synonyms-dictionary can be used for other models
            #print(len(words_of_interest),len(allSynonyms))

            allSynonyms = [tup for lst in brexitSynonyms for tup in lst]
            #print(len(allSynonyms)) 
            # For party factor alone =>Length should be 187*84=15708 OR len(words_of_interest)*len(mp-in-same-party pairs)
            # For party-time factor => Length should be 187*42=7854 OR len(w_of_int)*len(mp-in-same-party-same-time pairs)

            # Writing synonym files 
            # Change name for the pkl and txt files as per synonym-making factor, e.g. synonyms-party-time, etc

            with open(self.synPicklePath, 'wb') as f:
                pickle.dump(allSynonyms, f)

            with open(self.synTextPath,'w') as f:
                for tpl in allSynonyms:
                    for mptime in tpl:
                        f.write(mptime)
                        f.write(' ')
                    f.write('\n')
        else:
            self.logger.info('Retrofit Synonyms already created')

    def _retrofit_one_batch(self, syn_df_batch):

        this_logger = logging.getLogger(__name__)
        index_to_key = []
        result = np.zeros(
            shape=(len(syn_df_batch)*len(self.words_of_interest), self.vector_size)
        )

        this_logger.info(f'Processing index {syn_df_batch.index.start} to {syn_df_batch.index.stop}')
        index_count = 0
        # iterate over syn_df first because it takes time to load model.
        for row in syn_df_batch.itertuples():
            model = gensim.models.Word2Vec.load(row.full_model_path)
            if row.mpNamePartyInfo != 'dummy':
                for word in self.words_of_interest:
                    synonymString = f"{word}-{row.time}-{row.speaker.replace(' ','-')}-{row.mpNamePartyInfo}"
                    if word in model.wv.index_to_key:
                        result[index_count, :] = model.wv[word]
                        index_to_key.append(synonymString)
                        index_count += 1

        # POST PROCESSING:
        assert result[~np.all(result == 0, axis=1)].shape[0] == index_count
        assert index_count == len(index_to_key)
        result = result[~np.all(result == 0, axis=1)]

        this_logger.info(f'COMPLETE index {syn_df_batch.index.start} to {syn_df_batch.index.stop}')

        return index_to_key, result

    def _df_batch_generator(self, df, n):
        for i in range(0,df.shape[0],n):
            yield df[i:i+n]

    def retrofit_create_input_vectors(self, workers = None, overwrite=False):

        self.logger.info('Retrofit: Create Input Vectors')
        self.vectorFileName = os.path.join(self.retrofit_outdir,'vectorsPartyTime.hdf5')
        self.vectorIndexFileName = os.path.join(self.retrofit_outdir,f'vector_index_to_key_{self.parliament_name}.pkl')

        # sanity check that we do have retrofit savepaths readily accesible
        assert len(self.retrofit_model_paths) > 0

        if (os.path.isfile(self.vectorFileName) and overwrite) or not os.path.isfile(self.vectorFileName):

            # PREP DF
            with open(self.synPicklePath, 'rb') as f:
                synonyms = pickle.load(f)

            firstSyns = [tup[0] for tup in synonyms]
            secondSyns = [tup[1] for tup in synonyms]
            synonymsList = firstSyns+secondSyns
            uniqueSynonymsList = set(synonymsList)

            syn_df = pd.DataFrame(columns = ['full_model_path','modelKey', 'time', 'speaker', 'party'])
            syn_df['full_model_path'] = self.retrofit_model_paths
            syn_df['modelKey'] = [os.path.split(i)[-1] for i in self.retrofit_model_paths]
            syn_df['time'] = syn_df['modelKey'].apply(lambda x: x.split('df_')[1].split('_')[0])
            syn_df['speaker'] = syn_df['modelKey'].apply(lambda x: x.split('df_')[1].split('_')[1])
            syn_df['party'] = syn_df['speaker'].apply(lambda x: self.retrofit_prep_df[self.retrofit_prep_df['speaker'] == x]['party'].iat[0])

            # mpNamePartyInfo is meant to have stuff like '-Con' for Conservatives
            mpNames = []
            for row in syn_df.itertuples():
                # To ensure we don't match the likes of MP id 16 with MP id 216
                mpToSearch = f"-{row.speaker.replace(' ','-')}-"
                mpName='dummy'
                for syn in uniqueSynonymsList:
                    if(mpToSearch in syn):
                        mpName = syn.split(mpToSearch)[1]
                        break
                mpNames.append('default') if not mpName else mpNames.append(mpName)
            syn_df['mpNamePartyInfo'] = mpNames

            # retrieve required vector size from a file
            temp_model = gensim.models.Word2Vec.load(self.retrofit_model_paths[0])
            temp_vec = temp_model.wv[temp_model.wv.index_to_key[0]]
            self.vector_size = temp_vec.shape[0]

            # # open write stream to vector output file
            # with h5py.File(self.vectorFileName,'a') as f:
            #     g = f.require_dataset(
            #         self.parliament_name,
            #         shape=(len(self.words_of_interest)*len(syn_df), self.vector_size),
            #         dtype='float64',
            #         compression='lzf'
            #     )

            #     metadata_index_to_key = []
            #     index_count = 0
            #     for row in tqdm(syn_df.itertuples(), total=len(syn_df)):
            #         # iterate over syn_df first because it takes time to load model.
            #         model = gensim.models.Word2Vec.load(row.full_model_path)
            #         if row.mpNamePartyInfo != 'dummy':
            #             for word in tqdm(self.words_of_interest, leave=False):
            #                 synonymString = f"{word}-{row.time}-{row.speaker.replace(' ','-')}-{row.mpNamePartyInfo}"
            #                 if word in model.wv.index_to_key:
            #                     g[index_count, :] = model.wv[word]
            #                     metadata_index_to_key.append(synonymString)
            #                     index_count += 1
            write=True
            with h5py.File(self.vectorFileName,'a') as f:
                if self.parliament_name in f.keys() and overwrite:
                    del f[self.parliament_name]
                elif self.parliament_name in f.keys() and not overwrite:
                    write=False

            if not write:
                self.logger.info('Data already exists in hdf5 file, not generating new data')
            else:
                if workers is None:
                    workers = os.cpu_count()-1
                if workers > 1:
                    self.logger.info(f'Beginning Process Pool Executor with {workers} workers')
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        # results = list(tqdm(executor.map(self._retrofit_one_batch, self._df_batch_generator(syn_df, 100)), total=len(self._df_batch_generator(syn_df,100))))
                        results = executor.map(self._retrofit_one_batch, self._df_batch_generator(syn_df, 100))

                    # Combine results
                    index_to_key = []
                    total_result = np.array([])
                    for res in results:
                        index_to_key.extend(res[0])
                        if total_result.shape == (0,):
                            total_result = np.concatenate((total_result.reshape(0,res[1].shape[1]), res[1]), axis=0)
                        else:
                            total_result = np.concatenate((total_result, res[1]), axis=0)
                else:
                    index_to_key, total_result = self._retrofit_one_batch(syn_df)

                with h5py.File(self.vectorFileName, 'a') as f:
                    f.create_dataset(self.parliament_name, data=total_result, shape=total_result.shape)

                with open(self.vectorIndexFileName, 'wb') as f:
                    pickle.dump(index_to_key, f)

            # 2022-10-25 dev backup: save also as 

                        # # if(syn_df['mpNamePartyInfo'].iat[i]!='dummy'):
                        #     # if(word in syn_df['model'].iat[i].wv.index_to_key):

                        #         # synonymString = word+'-'+syn_df['time'].iat[i]+'-'+syn_df['mpId'].iat[i]+'-'+syn_df['mpNamePartyInfo'].iat[i]
                        #         wordVector = model.wv[word]
                        #         stringifiedVector = str(wordVector.flatten())

                        #         #The numpy array contains array brackets at the start and end, 
                        #         #This is not the format as in Faruqui's input code, hence replace
                        #         stringifiedVector = stringifiedVector.replace('[','').replace(']','')

                        #         #Strangely the vectors that start with a negative floating point have no space written 
                        #         #between the synonym key and the vector dimensions.
                        #         #So to check if the first dimension of the vector is <0 and if so, insert space before
                        #         stringVectorSplit = stringifiedVector.split()
                        #         if(stringVectorSplit[0]!=''):
                        #             if(float(stringVectorSplit[0])<0):
                        #                 stringifiedVector = ' '+stringifiedVector

                        #         f.write(synonymString)
                        #         f.write(stringifiedVector)

                        #         #To prevent writing an extra line break at the end of the file
                        #         if w_ind==len(self.words_of_interest)-1:
                        #             continue
                        #         else:
                        #             f.write('\n')
            return True
        else:
            self.logger.info('Retrofit: input vector creation already complete.')
            return None

    def retrofit_read_word_vecs_hdf5(self, dataset_key=None):
        """Read in vectors for retrofit from an hdf5 file. The original reading was from a text file which is a recipe for disaster.
        """

        wordVectors = {}
        if dataset_key is None:
            dataset_key = self.parliament_name
        with h5py.File(self.vectorFileName, 'r') as f:
            array = f[dataset_key][:]
        with open(self.vectorIndexFileName, 'rb') as f:
            index_to_key = pickle.load(f)

        #sanity check. actually length may be different
        # assert array.shape[0] == len(index_to_key)
        if array.shape[0] > len(index_to_key):
            print(f'Sanity check sum: {np.sum(array[index_to_key+1:])}')

        for word, vector in zip(index_to_key, array[:len(index_to_key)]):
            wordVectors[word] = vector

        return wordVectors

    def retrofit_output_vec(self, model_output_dir = None, overwrite=False):
        if (os.path.isfile(self.retrofit_outfile) and overwrite) or not os.path.isfile(self.retrofit_outfile):
            wordVecs = self.retrofit_read_word_vecs_hdf5()
            lexicon = retrofit.read_lexicon(self.synTextPath)
            numIter = int(10)
            self.retrofit_outfile = os.path.join(model_output_dir,'retrofit_out.txt')

            ''' Enrich the word vectors using ppdb and print the enriched vectors '''
            retrofit.print_word_vecs(retrofit.retrofit(wordVecs, lexicon, numIter), self.retrofit_outfile)
        else:
            self.logger.info(f'Retrofit: Retrofit file already exists at {self.retrofit_outfile}')

    def retrofit_post_process(self, change, no_change, model_output_dir):
        self.logger.info('Retrofit: Post Processing')

        with open(self.retrofit_outfile) as f:

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
        self.logger.info(str(len(vecs))+' Retrofitted vectors obtained')

        self.logger.info('Now extracting and mapping to synonym key')
        dictKeyVector = {}
        count=0
        for i in range(len(vecs)):

            vec = vecs[i].strip().split(' ')
            # Extracting synonym key
            synKey = vec[0]
            del(vec[0])
            vec=[i for i in vec if i!='']

            if(len(vec)!=300):
                self.logger.info('Vector with dimension<300', synKey,len(vec))
                count=count+1
            else:
                vec =[float(v) for v in vec]
                dictKeyVector[synKey]=vec
                npVec = np.array(dictKeyVector[synKey])
        self.logger.info('Count of vectors with fewer dimensions that we will not consider',count)
        dfRetrofitted = pd.DataFrame({'vectorKey':list(dictKeyVector.keys()), 'vectors':list(dictKeyVector.values())})

        # Filtering down words of interest as per those present in our vectors 
        # We're amending the computeAvgVec function accordingly
        # As it calculated based on processing from models, and here we're only taking vectors. Hence this check here too.

        vectorKeys =list(dfRetrofitted['vectorKey'])
        # Extracting words from vectors keys
        words_of_interest = list(set([vk.split('-')[0] for vk in vectorKeys]))
        # print(words_of_interest, len(words_of_interest))

        self.cosine_similarity_df = pd.DataFrame(columns = ('Word', 'Cosine_similarity'))

        # NOW WE ONLY HAVE THOSE WORDS HERE WHICH ARE PRESENT IN THE VECTORS.
        t1Keys = [t for t in list(dictKeyVector.keys()) if 't1' in t]
        t2Keys = [t for t in list(dictKeyVector.keys()) if 't2' in t]
        sims= []

        # Compute average of word in T1 and in T2 and store average vectors and cosine difference   
        for word in words_of_interest:

            #Provide a list of keys to average computation model for it to
            # #compute average vector amongst these models
            # wordT1Keys = [k for k in t1Keys if k.split('-')[0]==word]
            # wordT2Keys = [k for k in t2Keys if k.split('-')[0]==word]

            #Since here the key itself contains the word we're not simply sending T1 keys but sending word-wise key
            avgVecT1 = self.computeAvgVec(word, time = 't1', dictKeyVector = dictKeyVector)
            avgVecT2 = self.computeAvgVec(word, time = 't2', dictKeyVector = dictKeyVector)

            if(avgVecT1.shape == avgVecT2.shape):
                # Cos similarity between averages
                cosSimilarity = self.cosine_similarity(avgVecT1, avgVecT2)
                sims.append(cosSimilarity)
            else:
                self.logger.info('Word not found')
        self.cosine_similarity_df['Word']=words_of_interest
        self.cosine_similarity_df['Cosine_similarity']=sims

        '''
        self.cosine_similarity_df_sorted = self.cosine_similarity_df.sort_values(by='Cosine_similarity', ascending=True)
        self.cosine_similarity_df_sorted'''

        #Assigning change and no-change labels as initially expected
        self.cosine_similarity_df['semanticDifference']=['default' for i in range(self.cosine_similarity_df.shape[0])]
        self.cosine_similarity_df.loc[self.cosine_similarity_df['Word'].isin(change), 'semanticDifference'] = 'change' 
        self.cosine_similarity_df.loc[self.cosine_similarity_df['Word'].isin(no_change), 'semanticDifference'] = 'no_change'

        self.retrofit_dictkeyvector = dictKeyVector

        # Save into word2vec format for nn comparison
        for t in ['t1','t2']:
            self._save_word2vec_format(
                fname = os.path.join(model_output_dir, f'retrofit_vecs_{t}.bin'),
                vocab = dictKeyVector[t],
                vector_size = dictKeyVector[t][list(dictKeyVector[t].keys())[0]].shape[0]
            )

        self.model1 = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_output_dir, f'retrofit_vecs_t1.bin'), binary=True)
        self.model2 = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_output_dir, f'retrofit_vecs_t2.bin'), binary=True)

        self.logger.info('Retrofit: Post Process complete')

    def model(self, outdir, overwrite=False, min_vocab_size = 10000):
        """Function to generate the actual Word2Vec models.
        """
        self.outdir = outdir

        if self.model_type == 'whole':
            self.logger.info('MODELLING - WHOLE')

            savepath_t1 = os.path.join(outdir, 'whole_model_t1.model')
            savepath_t2 = os.path.join(outdir, 'whole_model_t2.model')

            if os.path.isfile(savepath_t1) and not overwrite:
                self.model1 = gensim.models.Word2Vec.load(savepath_t1)
                self.model2 = gensim.models.Word2Vec.load(savepath_t2)
                self.logger.info('MODELLING - whole models loaded in ')
            else:
                # create model for time 1
                self.logger.info('MODELLING - creating model for time 1')
                self.model1 = gensim.models.Word2Vec(self.data_t1['tokenized'])
                self.logger.info('MODELLING - creating model for time 2')
                self.model2 = gensim.models.Word2Vec(self.data_t2['tokenized'])

                self.model1.save(savepath_t1)
                self.model2.save(savepath_t2)

        if self.model_type == 'speaker':

            """
            With the speaker model, we train word embeddings for each MP and split into two time groups as well, t1 and t2.
            """

            self.split_speeches_by_mp = self.split_speeches(by='mp')
            self.speaker_saved_models = []
            for df_name, df in self.split_speeches_by_mp.items():
                model_savepath = os.path.join(outdir, f'speaker_{df_name}.model')
                if (os.path.isfile(model_savepath) and not overwrite):
                    if self.verbosity > 0:
                        self.logger.info('MODELLING - SPEAKER - Model exists and no overwrite flag set.')
                    self.speaker_saved_models.append(model_savepath)
                else:
                    try:
                        model = gensim.models.Word2Vec(df['Lemmas'])
                        model.save(model_savepath)
                        self.logger.info(f'MODELLING - SPEAKER - Saved model to {model_savepath}.')
                        if len(model.wv.index_to_key) < min_vocab_size:
                            continue
                        self.speaker_saved_models.append(model_savepath)
                    except:
                        if self.verbosity > 0:
                            self.logger.info(df.head())

        if self.model_type == 'retrofit':
            # create aligned models for retrofit.

            self.logger.info('MODELLING - RETROFIT')
            self.retrofit_model_paths = []
            new = False
            for row in self.retrofit_prep_df.itertuples(): 

                savepath = os.path.join(outdir, row.df_name)
                self.retrofit_model_paths.append(savepath)
                if (os.path.isfile(savepath) and overwrite) or not os.path.isfile(savepath):
                    new = True
                    model = gensim.models.Word2Vec(
                        row.Lemmas,
                        min_count=1,
                        vector_size=300,
                        window = 5,
                        sg = 1
                    )

                    # Skip if below minimum size
                    if len(model.wv.index_to_key) < min_vocab_size:
                        continue

                    # Previous: Also saving model in a dict and exporting
                    # Updated 22/10/22: save as you go along for RAM reasons. Also just better

                    model.save(savepath)

                    # dictOfModels[dframe] = model
                    #model.save(os.path.join(models_folder, modelName))

            if new:
                self.logger.info('MODELLING - RETROFIT - New models detected. Loading back in and running alignment')
                for ind, model_path in enumerate(self.retrofit_model_paths[:-1]):

                    model_current = gensim.models.Word2Vec.load(model_path)
                    check = np.array(model_current.wv[model_current.wv.index_to_key[0]])
                    model_next    = gensim.models.Word2Vec.load(self.retrofit_model_paths[ind+1])

                    _ = self._smart_procrustes_align_gensim(model_current, model_next)

                    if np.sum(check-model_current.wv[model_current.wv.index_to_key[0]])>0:
                        self.logger.warning('MODELLING - RETROFIT - PLEASE CHECK ALIGNMENT PROCEDURE')
                        return None

                    model_current.save(model_path)
                    model_next.save(self.retrofit_model_paths[ind+1])
                self.logger.info('MODELLING - RETROFIT - ALIGNMENT COMPLETE')
            else:
                self.logger.info('MODELLING - RETROFIT - NO NEW MODELS GENERATED -> NO ALIGNMENT NECESSARY')

    def cossim(self, word):
        sc = 1-spatial.distance.cosine(self.model1.wv[word], self.model2.wv[word])
        return sc

    def cosine_similarity(self, vec1, vec2):
        sc = 1-spatial.distance.cosine(vec1, vec2)
        return sc

    def computeAvgVec(self, w, time='t1', dictKeyVector = None):
        """Compute the average vector of a supplied word. Can be done from a dict of models containing the word or a dict of vectors directly.

        Args:
            w (str): word to compute average vector for
            time (str, optional): Either t1 or t2, indicating in a dict which models to compute average for. Defaults to 't1'.
            dictKeyVector (dict, optional): If provided, this will be the source of vectors to average. Overrides usage of dict of models. Should also have time provided. Defaults to None.

        Returns:
            avgEmbedding: Average embedding for the word
        """
        if dictKeyVector:
            modelsSum = np.zeros(np.array(dictKeyVector[list(dictKeyVector.keys())[0]]).shape[0])
        else:
            modelsSum = np.zeros(self.dictOfModels['t1'][0].layer1_size)

        # count for seeing how many to divide by
        count = 0
        if dictKeyVector:
            for key, value in dictKeyVector.items():
                if w in key and time in key:
                    modelsSum = np.add(modelsSum, value)
                    count += 1
        else:
            for model in self.dictOfModels[time]:
                try:
                    modelsSum = np.add(modelsSum, model.wv[w])
                    count += 1
                except KeyError:
                    continue

        if count == 0:
            if self.verbosity > 0:
                print(f'Word "{w}" not found')
            return modelsSum
        else:
            avgEmbedding = np.divide(modelsSum, count)
            return avgEmbedding

    def _save_word2vec_format(self, fname, vocab, vector_size, binary=True):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        vocab : dict
            The vocabulary of words.
        vector_size : int
            The number of dimensions of word vectors.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.

        FROM https://www.kaggle.com/code/matsuik/convert-embedding-dictionary-to-gensim-w2v-format/notebook

        """

        total_vec = len(vocab)
        with gensim.utils.open(fname, 'wb') as fout:
            print(total_vec, vector_size)
            fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, row in vocab.items():
                if binary:
                    row = row.astype(np.float32)
                    fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


    def woi(self, change, no_change):

        self.change = change
        self.no_change = no_change

        if self.model_type == 'whole':

            self.cosine_similarity_df = pd.DataFrame(([
                w,
                self.cossim(w),
                self.model1.wv.get_vecattr(w, "count"),
                self.model2.wv.get_vecattr(w, "count")
                ] for w in set(self.model1.wv.index_to_key).intersection(self.model2.wv.index_to_key)),
                columns = (
                    'Word',
                    'Cosine_similarity',
                    "Frequency_t1",
                    "Frequency_t2"
                )
            )

            self.cosine_similarity_df.loc[:,'FrequencyRatio'] = self.cosine_similarity_df['Frequency_t1']/self.cosine_similarity_df['Frequency_t2']
            self.cosine_similarity_df.loc[:,'TotalFrequency'] = self.cosine_similarity_df['Frequency_t1'] + self.cosine_similarity_df['Frequency_t2']

            self.cosine_similarity_df.sort_values(by='Cosine_similarity', ascending=True, inplace=True)

            self.words_of_interest = self.cosine_similarity_df[self.cosine_similarity_df['Word'].isin(change+no_change)].copy()

            self.words_of_interest.loc[self.words_of_interest['Word'].isin(change), 'semanticDifference'] = 'change'
            self.words_of_interest.loc[self.words_of_interest['Word'].isin(no_change), 'semanticDifference'] = 'no_change'

            print('words of interest complete')
            self.change_cossim = self.words_of_interest.loc[self.words_of_interest['semanticDifference'] == 'change', 'Cosine_similarity'] 
            self.no_change_cossim = self.words_of_interest.loc[self.words_of_interest['semanticDifference'] == 'no_change', 'Cosine_similarity'] 

            return self.words_of_interest, self.change_cossim, self.no_change_cossim

        elif self.model_type == 'speaker':

            self.words_of_interest = self.cosine_similarity_df[self.cosine_similarity_df['Word'].isin(change+no_change)]

            self.words_of_interest.loc[self.words_of_interest['Word'].isin(change), 'semanticDifference'] = 'change'
            self.words_of_interest.loc[self.words_of_interest['Word'].isin(no_change), 'semanticDifference'] = 'no_change'

            self.words_of_interest.sort_values(by='Cosine_similarity')

            print(self.words_of_interest)

            return self.words_of_interest

        elif self.model_type in ['retrofit', 'retro']:

            self.words_of_interest = change + no_change

    def postprocess(self, change_list, no_change_list, model_output_dir, workers=10, overwrite=False)->None:
        self.logger.info("POSTPROCESS: BEGIN")
        if self.model_type == 'speaker':
            self.process_speaker(model_output_dir)

        self.woi(change_list, no_change_list)

        if self.model_type in ['retrofit', 'retro']:
            self.retrofit_main_create_synonyms()
            self.retrofit_create_input_vectors(workers = workers, overwrite=overwrite)
            self.retrofit_output_vec(model_output_dir = model_output_dir)
            self.retrofit_post_process(change_list, no_change_list, model_output_dir)

    def logreg(self, model_output_dir, undersample = True):
        self.logger.info('RUNNING LOGREG')
        if self.model_type == 'retrofit':
            X = self.cosine_similarity_df['Cosine_similarity'].values.reshape(-1,1)
            y = self.cosine_similarity_df['semanticDifference']
            self.logger.info(self.cosine_similarity_df)
        else:
            X = self.words_of_interest['Cosine_similarity'].values.reshape(-1,1)
            y = self.words_of_interest['semanticDifference']
            self.logger.info(self.words_of_interest)

        if undersample:
            undersample = RandomUnderSampler(sampling_strategy=1.0)

            X_over, y_over = undersample.fit_resample(X, y)
            X, y = X_over, y_over

        # CHANGE_PROPORTION = np.sum(y == 'change')/len(y)
        CHANGE_PROPORTION = 0.25
        stratification = np.random.choice(['change','no_change'],size=(len(y),), p=[CHANGE_PROPORTION, 1-CHANGE_PROPORTION])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=stratification)

        self.logger.info(f'Y value counts: {y.value_counts()}')
        self.logger.info(f'Y train value counts: {y_train.value_counts()}')

        logreg = LogisticRegression()
        kf = logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)


        scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,pos_label='change'),
               'recall' : make_scorer(recall_score,pos_label='change'), 
               'f1_score' : make_scorer(f1_score,pos_label='change')}

        scores = cross_validate(kf, X, y, cv=min(10, len(y)), scoring=scoring,error_score='raise')
        accuracy, precision, recall, f1_score_res = [], [], [], []

        self.logger.info('Accuracy', scores['test_accuracy'].mean())
        self.logger.info('Precision', scores['test_precision'].mean())
        self.logger.info('Recall', scores['test_recall'].mean())
        self.logger.info('F1 Score', scores['test_f1_score'].mean())

        accuracy.append(scores['test_accuracy'].mean())
        precision.append(scores['test_precision'].mean())
        recall.append(scores['test_recall'].mean())
        f1_score_res.append(scores['test_f1_score'].mean())

        scoresDict = {
            'Model':['whole Model'],
            'Basis': ['Cosine Similarity'],
            'Accuracy':accuracy,
            'Precision':precision,
            'Recall':recall,
            'F1Score':f1_score
        }
        scoresDf = pd.DataFrame(scoresDict)
        self.logger.info(scoresDf)
        #save result
        scoresDf.to_csv(os.path.join(model_output_dir, 'logreg.csv'))

    def nn_comparison(self, model_output_dir, undersample = True):
        print('\n Running Nearest Neighbours Comparison')
        neighboursInT1 = []
        neighboursInT2 = []

        if self.model_type in ['retrofit', 'retro']:
            self.words_of_interest = self.cosine_similarity_df

        for word in self.words_of_interest['Word'].to_list():

            if self.model_type in ['speaker', 'retrofit', 'retro']:
                x = self.model1.similar_by_word(word,10)
                y = self.model2.similar_by_word(word,10)
            elif self.model_type == 'whole':
                x = self.model1.wv.similar_by_word(word,10) 
                y = self.model2.wv.similar_by_word(word,10)

            x = [tup[0] for tup in x]
            y = [tup[0] for tup in y]

            neighboursInT1.append(x)
            neighboursInT2.append(y)

        self.words_of_interest['neighboursInT1'] = neighboursInT1
        self.words_of_interest['neighboursInT2'] = neighboursInT2

        #words_of_interest['overlappingNeighbours'] = ?
        #intersectingNeighbs = set(words_of_interest['neighboursInT1'].to_list()).intersect(words_of_interest['neighboursInT2'].to_list())
        lengthOverlap = []

        for index in (self.words_of_interest['neighboursInT1'].index):
            neighboursT1 = self.words_of_interest.at[index, 'neighboursInT1']
            neighboursT2 = self.words_of_interest.at[index, 'neighboursInT2']
            lengthOverlap.append(len(set(neighboursT1).intersection(neighboursT2)))

        self.words_of_interest['overlappingNeighbours'] = lengthOverlap

        self.words_of_interest[self.words_of_interest['semanticDifference']=='change']['overlappingNeighbours'].describe()
        self.words_of_interest[self.words_of_interest['semanticDifference']=='no_change']['overlappingNeighbours'].describe()
        neighbours_of_changed_words = self.words_of_interest[self.words_of_interest['semanticDifference']=='change'].sort_values(by='Cosine_similarity',ascending=True)[['Word','neighboursInT1','neighboursInT2']]

        X = self.words_of_interest['overlappingNeighbours'].values.reshape(-1,1)
        y = self.words_of_interest['semanticDifference']

        if undersample:
            undersample = RandomUnderSampler(sampling_strategy=1.0)

            X_over, y_over = undersample.fit_resample(X, y)
            X=X_over
            y=y_over

        # CHANGE_PROPORTION = np.sum(y == 'change')/len(y)
        CHANGE_PROPORTION = 0.25
        stratification = np.random.choice(['change','no_change'],size=(len(y),), p=[CHANGE_PROPORTION, 1-CHANGE_PROPORTION])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=stratification)

        self.logger.info(f'Y value counts: {y.value_counts()}')
        self.logger.info(f'Y train value counts: {y_train.value_counts()}')

        logreg = LogisticRegression()
        kf = logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)


        scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,pos_label='change'),
               'recall' : make_scorer(recall_score,pos_label='change'), 
               'f1_score' : make_scorer(f1_score,pos_label='change')}

        scores = cross_validate(kf, X, y, cv=min(10, len(y)), scoring=scoring,error_score='raise')
        accuracy, precision, recall, f1_score_res = [], [], [], []

        self.logger.info('Accuracy', scores['test_accuracy'].mean())
        self.logger.info('Precision', scores['test_precision'].mean())
        self.logger.info('Recall', scores['test_recall'].mean())
        self.logger.info('F1 Score', scores['test_f1_score'].mean())

        accuracy.append(scores['test_accuracy'].mean())
        precision.append(scores['test_precision'].mean())
        recall.append(scores['test_recall'].mean())
        f1_score_res.append(scores['test_f1_score'].mean())

        scoresDict = {'Model':['whole Model'],'Basis': ['Cosine Similarity'],'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1Score':f1_score}
        scoresDf = pd.DataFrame(scoresDict)
        scoresDf

        group1= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'change']
        group2= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'no_change']

        # T Test with 10 neighbours

        summary_neighbours, results_neighbours = rp.ttest(group1= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'change'], group1_name= "change",
                                    group2= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'no_change'], group2_name= "no_change")
        # print(summary_neighbours)
        summary_neighbours.to_csv(os.path.join(model_output_dir, 'nn_comparison.csv'))



@click.command()
@click.option('--file', '-f', required=True, help='File')
@click.option('--change', '-c', required=True, help='Text file containing words expected to have changed', type=click.File())
@click.option('--no_change', '-nc', required=False, help='Text file containing words NOT expected to have changed', type=click.File())
@click.option('--outdir', required=True, help='Output file directory')
@click.option('--model_output_dir', required=True, help='Outputs after model generation, such as average vectors')
@click.option('--model', required=False, default='whole')
@click.option('--tokenized_outdir', required=False)
@click.option('--min_vocab_size', required=False, type=int)
@click.option('--split_date', required=False, default='2016-06-23 23:59:59')
@click.option('--split_range', required=False)
@click.option('--retrofit_outdir', required=False)
@click.option('--undersample', required=False, is_flag = True)
@click.option('--log_level', required=False, default='INFO')
@click.option('--log_dir', required=False)
@click.option('--log_handler_level', required=False, default='stream')
@click.option('--overwrite_preprocess', required=False, is_flag=True)
@click.option('--overwrite_model', required=False, is_flag=True)
@click.option('--overwrite_postprocess', required=False, is_flag=True)
def main(
        file,
        change,
        no_change,
        outdir,
        model_output_dir,
        tokenized_outdir,
        min_vocab_size,
        split_date,
        split_range,
        retrofit_outdir,
        model,
        undersample,
        log_level,
        log_dir,
        log_handler_level,
        overwrite_preprocess,
        overwrite_model,
        overwrite_postprocess
    ):
    """Semantic Change.

    Args:
        file (csv): raw data file.
        change (_type_): _description_
        no_change (_type_): _description_
        outdir (_type_): _description_
        model_output_dir (_type_): _description_
        tokenized_outdir (_type_): _description_
        retrofit_outdir (_type_): _description_
        model (str, optional): _description_.
    """

    logging_dict = {
            'NONE': None,
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
    }

    logging_level = logging_dict[log_level]

    if logging_level is not None:

        logging_fmt   = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
        # today_datetime = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        if log_dir is not None:
            assert os.path.isdir(log_dir)
            logging_file  = os.path.join(log_dir, f'retrofit.log')

        if log_handler_level == 'both':
            handlers = [
                logging.FileHandler(filename=logging_file,mode='a'),
                logging.StreamHandler()
            ]
        elif log_handler_level == 'file':
            handlers = [logging.FileHandler(filename=logging_file,mode='a')]
        elif log_handler_level == 'stream':
            handlers = [logging.StreamHandler()]
        logging.basicConfig(
            handlers=handlers,
            format=logging_fmt,
            level=logging_level,
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logger = logging.getLogger(__name__)

    # Set so gensim doesn't go crazy with the logs
    logging.getLogger('gensim.utils').setLevel(logging.DEBUG)
    logging.getLogger('gensim.utils').propagate = False
    logging.getLogger('gensim.models.word2vec').setLevel(logging.DEBUG)
    logging.getLogger('gensim.models.word2vec').propagate = False

    # Log all the parameters
    logger.info(f'PARAMS - file - {file}')
    logger.info(f'PARAMS - change - {change}')
    logger.info(f'PARAMS - no_change - {no_change}')
    logger.info(f'PARAMS - outdir - {outdir}')
    logger.info(f'PARAMS - min_vocab_size - {min_vocab_size}')
    logger.info(f'PARAMS - split date -  {split_date}')
    logger.info(f'PARAMS - split range - {split_range}')
    logger.info(f'PARAMS - model_output_dir - {model_output_dir}')
    logger.info(f'PARAMS - tokenized_outdir - {tokenized_outdir}')
    logger.info(f'PARAMS - retrofit_outdir - {retrofit_outdir}')
    logger.info(f'PARAMS - model - {model}')

    # process change lists
    change_list = []
    for i in change:
        change_list.append(i.strip('\n'))
    no_change_list = []
    if no_change:
        for i in no_change:
            no_change_list.append(i.strip('\n'))

    # instantiate parliament data handler
    handler = ParliamentDataHandler.from_csv(file, tokenized=False)
    handler.tokenize_data(tokenized_data_dir = tokenized_outdir, overwrite = False)
    date_to_split = split_date
    logger.info(f'SPLITTING BY DATE {date_to_split}')
    handler.split_by_date(date_to_split, split_range)

    # unified
    handler.preprocess(
        model = model,
        model_output_dir = model_output_dir,
        retrofit_outdir=retrofit_outdir,
        overwrite=overwrite_preprocess
    )
    handler.model(
        outdir,
        overwrite=overwrite_model,
        min_vocab_size=min_vocab_size
    )
    handler.postprocess(
        change_list,
        no_change_list,
        model_output_dir,
        workers = 10,
        overwrite=overwrite_postprocess
    )
    handler.logreg(model_output_dir, undersample)
    handler.nn_comparison(model_output_dir, undersample)

if __name__ == '__main__':
    main()