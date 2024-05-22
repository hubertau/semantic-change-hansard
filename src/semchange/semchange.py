"""
Given input data, lemmatize and get metadata
"""

import csv
import datetime
import functools
import multiprocessing
import logging
import os
from collections import defaultdict, Counter
import pickle
from concurrent.futures import ProcessPoolExecutor
from csv import reader

import gc
import re
import click
from dateutil.relativedelta import relativedelta
import gensim
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

import h5py
import imblearn
import matplotlib.pyplot as plt
import nltk
import re
import numpy as np
import pandas as pd
import researchpy as rp
import spacy
from itertools import product
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
from dataclasses import dataclass

@dataclass
class syn_identifier:
    word: str
    party: str
    time: str = None
    debate: str = None

    def stringify(self) -> str:
        valid_parts = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                valid_parts.append(value)
        return "_".join(valid_parts)

@dataclass
class synonym_item:
    """Class for keeping track of a synonym item in retrofitting."""
    word: str
    time: str
    speaker: str
    party: str
    debate: str = None

    def stringify(self) -> str:
        valid_parts = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                valid_parts.append(value)
        return "$".join(valid_parts)

    @classmethod
    def from_string(cls, input_string):
        i = input_string.split('$')
        assert len(i) >= 4
        return cls(*i)


class ParliamentDataHandler(object):

    def __init__(self, data, tokenized, data_filename = None, verbosity=0):
        self.data = data
        self.tokenized = tokenized
        self.split_complete = False
        self.data_filename = data_filename
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_csv(cls, data, tokenized=False):
        df = pd.read_csv(data)
        return cls(df, tokenized=tokenized, data_filename=data)

    def _break_into_sentences(self, paragraph):
        if isinstance(paragraph, str):
            # Split the paragraph at every '.' or '?', but capture the delimiters
            sentences = re.split(r'(\.|\?)', paragraph)
            # Combine the sentences with their respective delimiters
            sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') for i in range(0, len(sentences), 2)]
            # Remove any empty strings from the list, which can occur after splitting.
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        elif isinstance(paragraph, float):
            self.logger.warning(f"Float detected in tokenization: {paragraph}")
            sentences = [paragraph]
        elif isinstance(sentence, list):
            self.logger.warning(f"List detected in tokenization: {paragraph}")
            sentences = paragraph
        return sentences

    def _tokenize_one_sentence(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.lower()
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sentence)
        elif isinstance(sentence, float):
            self.logger.warning(f"Float detected in tokenization: {sentence}")
            tokens = [sentence]
        elif isinstance(sentence, list):
            self.logger.warning(f"List detected in tokenization: {sentence}")
            tokens = sentence
        return tokens

    def _tokenize_each_sentence(self, sentences):
        return [ self._tokenize_one_sentence(sentence) for sentence in sentences ]

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
            # Break into sentences
            sentences = self.data.text.apply(self._break_into_sentences)
            self.data['sentences'] = sentences
            # Also save the words present in every sentence
            self.data['tokenized_sentences'] = self.data.sentences.apply(self._tokenize_each_sentence)
            # And tokenize
            #tokens = self.data.text.apply(self._tokenize_one_sentence)
            self.data['tokenized'] = self.data['tokenized_sentences'].apply(lambda x: [j for i in x for j in i])

            self.data.to_pickle(savepath)
            self.logger.info(f'Saved to {savepath}')
        elif (os.path.isfile(savepath) and not overwrite):
            self.logger.info(f'Loading in tokenized data from {savepath}')
            self.data = pd.read_pickle(savepath)

    def split_by_date(self, date, split_range):

        # conert relevant stuff into datetime objects
        date = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
        if split_range is not None:
            leftbound  = date - relativedelta(years=split_range)
            rightbound = date + relativedelta(years=split_range)
        else:
            leftbound  = date - relativedelta(years=100)
            rightbound = date + relativedelta(years=100)

        # convert date column to datetime before we do any more processing
        self.data['date'] = pd.to_datetime(self.data['date'])

        # make sure we have a column... lol
        assert 'date' in self.data.columns

        # do the splitting
        self.data.loc[(self.data['date'] > leftbound) & (self.data['date'] <= date), 'time'] = 't1'
        self.data.loc[(self.data['date'] > date) & (self.data['date'] <= 
        rightbound), 'time'] = 't2'

        # also restrict range to the years we want.
        if split_range is not None:
            to_discard = self.data['time'].isnull()
            self.logger.debug(f'Row to discard due to time split: {sum(to_discard)} out of {len(self.data)} = {100*sum(to_discard)/len(self.data):.2f}')
            self.data = self.data[~to_discard]
            self.data = self.data.reset_index()
            assert len(self.data['time'].unique()) == 2

        # now hash the debate ids for retrofit if we are doing that
        self.data.loc[:,'debate_id'] = self.data['agenda'].map(hash)
        self.data.loc[:,'debate'] = self.data['agenda']

        # ensure data integrity of speaker column
        self.data = self.data[self.data['speaker'].apply(lambda x: isinstance(x,str))]

        # ensure data integrity of party column
        self.data = self.data[self.data['party'].apply(lambda x: isinstance(x,str))]

        # let's just get rid of spaces so there's no weird stuff
        self.data.loc[:,'speaker'] = self.data['speaker'].apply(lambda x: x.replace(' ', '_'))

        # same for parties
        self.data.loc[:,'party'] = self.data['party'].apply(lambda x: x.replace(' ', '_'))

        # generate set of tokens for each row for O(1) comparison later for checking membership
        self.data.loc[:, 'token_set'] = self.data['tokenized'].apply(set)

        # split the data into two parts for ease of modelling in the whole model, say
        self.data_t1 = self.data[(self.data['date'] > leftbound) & (self.data['date'] <= date)]
        self.data_t2 = self.data[(self.data['date'] > date) & (self.data['date'] < rightbound)]

        # verbose debug output
        self.logger.debug(f'Data t1 len: {len(self.data_t1)}')
        self.logger.debug(f'Data t2 len: {len(self.data_t2)}')

        # for any referencing later
        self.split_complete = True

    def preprocess(self, change = None, no_change = None, model = None, model_output_dir = None, retrofit_outdir=None, overwrite=None):
        """TODO: Use this function to unify the retrofit prep, the tokenising, splitting of speeches, etc. so this is not duplicated in subsequent processes"""
        assert model in ['retrofit', 'retro', 'whole', 'speaker', 'speaker_plus']
        self.change = change
        self.no_change = no_change
        self.model_type = model
        if self.model_type in ['retrofit', 'retro']:
            self.logger.info(f'PREPROCESS: Running preprocessing for retrofit.')
            self.retrofit_prep(retrofit_outdir=retrofit_outdir, overwrite = overwrite)
        elif self.model_type in ['whole', 'speaker', 'speaker_plus']:
            self.logger.info(f'PREPROCESS - {self.model_type.upper()} - None required.')

    def _get_retrofit_word_counts(self, word_list, time='t1'):
        output = Counter()
        for model_path in self.retrofit_model_paths:
            if time in model_path:
                model = gensim.models.Word2Vec.load(model_path)
                for word in word_list:
                    try:
                        output[word] += model.wv.get_vecattr(word, 'count')
                    except:
                        continue
        return output

    def retrofit_prep(self, retrofit_outdir=None, overwrite=False):
        if retrofit_outdir:
            assert os.path.isdir(retrofit_outdir)
            self.retrofit_outdir = retrofit_outdir
        if 'parliament' in self.data.columns:
            self.parliament_name = self.data['parliament'].iat[0]
        else:
            self.parliament_name = 'UNKNOWN PARLIAMENT'

        retrofit_prep_savepath = os.path.join(retrofit_outdir, f'retrofit_prep.json')
        self.logger.info(f'Retrofit Prep Path is {retrofit_prep_savepath}')

        self.logger.info('Running retrofit prep')
        if os.path.isfile(retrofit_prep_savepath) and overwrite or not os.path.isfile(retrofit_prep_savepath):

            self.logger.info('Splitting speeches')

            total_mpTimedf = self.split_speeches_df(by='speaker')
            total_mpTimedf['tokens_delist'] = [','.join(map(str, l)) for l in total_mpTimedf['tokens']]

            total_mpTimedf['LengthTokens'] = total_mpTimedf.tokens.map(len)

            total_mpTimedf.to_json(retrofit_prep_savepath, orient='split', index=False)

            self.retrofit_prep_df = total_mpTimedf

            self.logger.info(f'Retrofit prep saved to {retrofit_prep_savepath}')

        else:

            self.logger.info('Loading in prep from before...')

            self.retrofit_prep_df = pd.read_json(retrofit_prep_savepath, orient='split')
            self.logger.info(f'Retrofit prep loaded in from {retrofit_prep_savepath}, key = {self.parliament_name}')

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

    def split_speeches_df(self, by='party'):
        """Function to split loaded data by party

        Returns:
            dictSpeechesbyParty (dict): dict of partty:
        """

        self.logger.info('Splitting speeches...')


        if by == 'party':
            splitspeeches = pd.DataFrame(columns=['df_name' ,'tokens', 'party'])
        elif by == 'mp' or by == 'speaker':
            by = 'speaker'
            splitspeeches = pd.DataFrame(columns=[
                'df_name',
                by ,
                'tokens',
                'party',
                'debate',
                'debate_id'
                ]
            )
        # self.data_t1['time'] = 't1'
        # self.data_t2['time'] = 't2'
        split_t1 = list(self.data_t1[by].unique())
        split_t2 = list(self.data_t2[by].unique())
        total_split = set(split_t1+split_t2)


        for p in total_split:

            for dfTime in ['df_t1','df_t2']:

                tempDf = pd.DataFrame()
                tempList = []
                dfName = f"{dfTime}_{p}"

                if (dfTime == 'df_t1'):
                    tempDf = self.data_t1[self.data_t1[by]==p].copy()

                elif (dfTime == 'df_t2'):
                    tempDf = self.data_t2[self.data_t2[by]==p].copy()

                if (tempDf.shape[0]==0):
                    continue

                # used to index 'Lemmas'
                tempDf.loc[:,'tokens'] = tempDf['tokenized']
                tempList.extend(tempDf['tokens'].to_list())
                split = tempDf[by].iat[0]
                party = tempDf['party'].iat[0]
                debate = tempDf['agenda'].to_list()
                debate_ids = tempDf['debate_id'].to_list()

                #Flatten the list so it's not a list of lists
                # 2022-10-10: This is the offending line that splits words into letters screwing up the subsequent stuff.
                # tempList = [item for sublist in tempList for item in sublist]

                tempDf = pd.DataFrame([
                    [split, tempList, party, debate, debate_ids]],
                    columns=[by, 'tokens', 'party', 'debate', 'debate_id']
                )
                tempDf['df_name'] = dfName
                splitspeeches = pd.concat([splitspeeches, tempDf], axis=0)
                # splitspeeches[dfName]= tempDf
                # splitspeeches[dfName]['df_name'] = dfName

        return splitspeeches

    def model(self,
        outdir,
        embedding = 'word',
        overwrite=False,
        skip_model_check = False,
        min_vocab_size = 10000,
        overlap_req = 0.5,
        align = True
    ):
        """Function to generate the actual word/sentence embedding models.
        """
        self.outdir = outdir

        if self.embedding=='word':
            self.logger.info('MODELLING - WORD EMBEDDINGS')

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
                    self.model1 = gensim.models.Word2Vec(
                            self.data_t1['tokenized'],
                            min_count=1,
                            vector_size=300,
                            window = 5,
                            sg = 1
                        )
                    self.logger.info('MODELLING - creating model for time 2')
                    self.model2 = gensim.models.Word2Vec(
                            self.data_t2['tokenized'],
                            min_count=1,
                            vector_size=300,
                            window = 5,
                            sg = 1
                        )

                    _ = self._smart_procrustes_align_gensim(self.model1, self.model2)

                    common_vocab = len(set(self.model1.wv.index_to_key).intersection(set(self.model2.wv.index_to_key)))
                    self.logger.info('MODELLING - WHOLE - ALIGNMENT COMPLETE')
                    if common_vocab == 0:
                        self.logger.error("MODELLING - WHOLE - NO COMMON VOCAB LEFT OVER")

                    self.model1.save(savepath_t1)
                    self.model2.save(savepath_t2)

            if self.model_type in ['speaker', 'speaker_plus']:

                """
                With the speaker model, we train word embeddings for each MP and split into two time groups as well, t1 and t2.
                """

                self.split_speeches_by_mp = self.split_speeches_df(by='speaker')
                self.speaker_saved_models = []
                new = False
                count = 0
                skipped = 0
                vocab_skipped = 0
                for row in self.split_speeches_by_mp.itertuples():
                    model_savepath = os.path.join(outdir, f'speaker_{row.df_name}.model')

                    # if the file exists and we are not overwriting, record that is is there.
                    if (os.path.isfile(model_savepath) and not overwrite):
                        self.logger.info(f'{os.path.split(model_savepath)[-1]} found and not overwriting...')
                        self.speaker_saved_models.append(model_savepath)

                    # if the file does not exist and we are not checking if it should, just continue
                    elif (not os.path.isfile(model_savepath)) and skip_model_check:
                        self.logger.info(f'{os.path.split(model_savepath)[-1]} not found but skipping model check...')
                        continue

                    # otherwise, generate model
                    else:
                        try:
                            count += 1
                            model = gensim.models.Word2Vec(
                                row.tokens,
                                min_count=1,
                                vector_size=300,
                                window = 5,
                                sg = 1
                            )
                            vocab_of_interest = set(self.change)
                            req_size = len(vocab_of_interest)
                            overlap = len(vocab_of_interest.intersection(set(model.wv.index_to_key)))/req_size
                            if overlap<overlap_req:
                                vocab_skipped += 1
                                self.logger.info(f'MODELLING - Skipped {row.df_name} due to not enough overlap with words of interest. Overlap: {overlap:.2f}')
                                continue
                            # Skip if below minimum size
                            if len(model.wv.index_to_key) < min_vocab_size:
                                skipped += 1
                                self.logger.info(f'MODELLING - SKIPPED {row.df_name} due to insufficient vocab size. Vocab size: {len(model.wv.index_to_key)}')
                                continue
                            if not new:
                                new = True
                                self.logger.info('New set to True')
                            model.save(model_savepath)
                            if count % 100 == 0:
                                self.logger.info(f'MODELLING - {count}/{len(self.split_speeches_by_mp)} = {100*count/len(self.split_speeches_by_mp):.2f}% complete')
                            # self.logger.info(f'MODELLING - SPEAKER - Saved model to {model_savepath}.')
                            self.speaker_saved_models.append(model_savepath)
                        except Exception as e:
                            self.logger.error(e)

                self.logger.info(f"num speaker models: {len(self.speaker_saved_models)}")
                if count > 0:
                    self.logger.info(f"MODELLING - SPEAKER - {skipped} out of {count}  models skipped due to vocab size")
                    self.logger.info(f"MODELLING - SPEAKER - {vocab_skipped} out of {count} models skipped due to insufficient overlap with vocab of interest")
                    self.logger.info(f"Total skipped: {vocab_skipped+skipped} out of {count} = {100*(vocab_skipped+skipped)/count:.2f}%")
                else:
                    self.logger.info(f"No models made or skipped")

                if new and align:
                    self.logger.info('MODELLING - SPEAKER - New models detected. Loading back in and running alignment')
                    for ind, model_path in enumerate(self.speaker_saved_models[:-1]):

                        model_current = gensim.models.Word2Vec.load(model_path)
                        check = np.array(model_current.wv[model_current.wv.index_to_key[0]])
                        model_next    = gensim.models.Word2Vec.load(self.speaker_saved_models[ind+1])

                        _ = self._smart_procrustes_align_gensim(model_current, model_next)

                        current_common_vocab_size = len(set(model_current.wv.index_to_key).intersection(set(model_next.wv.index_to_key)))
                        self.logger.info(f"MODELLING - SPEAKER - ALIGNMENT - CURRENT COMMON VOCAB IS {current_common_vocab_size} after alignment at index {ind}, model: {model_path}")

                        if np.sum(check-model_current.wv[model_current.wv.index_to_key[0]])>0:
                            self.logger.warning('MODELLING - SPEAKER - PLEASE CHECK ALIGNMENT PROCEDURE')
                            return None

                        model_current.save(model_path)
                        model_next.save(self.speaker_saved_models[ind+1])
                    self.logger.info('MODELLING - SPEAKER - ALIGNMENT COMPLETE')
                    if current_common_vocab_size == 0:
                        self.logger.error("MODELLING - SPEAKER - NO COMMON VOCAB LEFT OVER")
                else:
                    self.logger.info('MODELLING - SPEAKER - NO NEW MODELS GENERATED -> NO ALIGNMENT NECESSARY')

            if self.model_type == 'retrofit':
                # create aligned models for retrofit.

                self.logger.info('MODELLING - RETROFIT')
                self.retrofit_model_paths = []
                new = False
                count = 0
                skipped = 0
                vocab_skipped = 0
                for row in self.retrofit_prep_df.itertuples(): 

                    savepath = os.path.join(outdir, f"{row.df_name}.model")
                    if (os.path.isfile(savepath) and overwrite) or (not os.path.isfile(savepath) and not skip_model_check):
                        model = gensim.models.Word2Vec(
                            row.tokens,
                            min_count=1,
                            vector_size=300,
                            window = 5,
                            sg = 1
                        )

                        count += 1
                        # Skip if not containing correct vocab
                        vocab_of_interest = set(self.change)
                        req_size = len(vocab_of_interest)
                        overlap = len(vocab_of_interest.intersection(set(model.wv.index_to_key)))/req_size
                        if overlap<overlap_req:
                            vocab_skipped += 1
                            self.logger.debug(f'Modelling: Skipped {row.df_name} due to not enough overlap with words of interest. Overlap: {overlap:.2f}')
                            continue
                        # Skip if below minimum size
                        if len(model.wv.index_to_key) < min_vocab_size:
                            skipped += 1
                            self.logger.debug(f'MODELLING - SKIPPED {row.df_name} due to insufficient vocab size. Vocab size: {len(model.wv.index_to_key)}')
                            continue

                        if not new:
                            new = True
                            self.logger.info('New set to True')

                        if count % 100 == 0:
                            self.logger.info(f'MODELLING - {count}/{len(self.retrofit_prep_df)} = {100*count/len(self.retrofit_prep_df):.2f}% complete')

                        # N.B. only append savepath if retrofit model satisfies criterion.
                        self.retrofit_model_paths.append(savepath)

                        # Previous: Also saving model in a dict and exporting
                        # Updated 22/10/22: save as you go along for RAM reasons. Also just better

                        model.save(savepath)

                        # dictOfModels[dframe] = model
                        #model.save(os.path.join(models_folder, modelName))
                    elif os.path.isfile(savepath):
                        self.retrofit_model_paths.append(savepath)
                        count += 1

                self.logger.info(f"MODELLING - RETROFIT - {skipped} out of {count}  models skipped due to vocab size")
                self.logger.info(f"MODELLING - RETROFIT - {vocab_skipped} out of {count} models skipped due to insufficient overlap with vocab of interest")
                self.logger.info(f"Total skipped: {vocab_skipped+skipped} out of {count} = {100*(vocab_skipped+skipped)/count:.2f}%")

                if new and align:
                    self.logger.info('MODELLING - RETROFIT - New models detected. Loading back in and running alignment')
                    for ind, model_path in enumerate(self.retrofit_model_paths[:-1]):

                        model_current = gensim.models.Word2Vec.load(model_path)
                        check = np.array(model_current.wv[model_current.wv.index_to_key[0]])
                        model_next    = gensim.models.Word2Vec.load(self.retrofit_model_paths[ind+1])

                        _ = self._smart_procrustes_align_gensim(model_current, model_next)

                        current_common_vocab_size = len(set(model_current.wv.index_to_key).intersection(set(model_next.wv.index_to_key)))
                        self.logger.debug(f"MODELLING - ALIGNMENT - CURRENT COMMON VOCAB IS {current_common_vocab_size} after alignment at index {ind}, model: {model_path}")

                        if np.sum(check-model_current.wv[model_current.wv.index_to_key[0]])>0:
                            self.logger.warning('MODELLING - RETROFIT - PLEASE CHECK ALIGNMENT PROCEDURE')
                            return None

                        model_current.save(model_path)
                        model_next.save(self.retrofit_model_paths[ind+1])
                    self.logger.info('MODELLING - RETROFIT - ALIGNMENT COMPLETE')
                    if current_common_vocab_size == 0:
                        self.logger.error("MODELLING - RETROFIT - NO COMMON VOCAB LEFT OVER")
                else:
                    self.logger.info('MODELLING - RETROFIT - NO NEW MODELS GENERATED -> NO ALIGNMENT NECESSARY')


        elif self.embedding=='sentence':
            self.logger.info('MODELLING - SENTENCE EMBEDDINGS')

            if self.model_type == 'whole':
                self.logger.info('MODELLING - WHOLE')

                savepath_t1 = os.path.join(outdir, 'whole_model_t1.model.sbert')
                savepath_t2 = os.path.join(outdir, 'whole_model_t2.model.sbert')

                if os.path.isfile(savepath_t1) and not overwrite:
                    self.model1 = gensim.models.Word2Vec.load(savepath_t1)
                    self.model2 = gensim.models.Word2Vec.load(savepath_t2)
                    self.logger.info('MODELLING - whole models loaded in ')
                else:
                    # create model for time 1
                    self.logger.info('MODELLING - creating model for time 1')

                    self.model = SentenceTransformer('all-MiniLM-L6-v2')

                    # Get the sentence embedding for all sentences
                    d = self.data_t1
                    embeddings1 = d.sentences.apply(self.model.encode) 
                    
                    all_words_t1 = set().union(*d["tokenized"])
                    
                    # Build dictionary of words-to-idx-of-embeddings-of-where-they-appear
                    words2embeddingidx = { w:[] for w in all_words_t1 }
                    for (idx_row, row), embeddings in zip(d.iterrows(), embeddings1):
                        for idx_sentence, tokenized_sentence in enumerate(row['tokenized_sentences']):
                            for word in tokenized_sentence:
                                words2embeddingidx[word].append( (idx_row, idx_sentence)) 

                    # Use that dictionary to get the sentence embedding for each context where the word has been used (per time, per speaker, etc)
                    words2embeddings1 = { word:[ embeddings1.loc[idx_row][idx_sentence] for (idx_row,idx_sentence) in idxs]
                                                                                         for word,idxs in words2embeddingidx.items() }

                    # Take the centroid of those sentence embeddings, make a dictionary of words and their centroids
                    words2centroids1  = { word:np.mean(embeddings,axis=0) for word,embeddings in words2embeddings1.items() }

                    # create model for time 2
                    self.logger.info('MODELLING - creating model for time 2')

                    self.model = SentenceTransformer('all-MiniLM-L6-v2')

                    # Get the sentence embedding for all sentences
                    d = self.data_t2
                    embeddings2 = d.sentences.apply(self.model.encode) 
                    
                    all_words_t2 = set().union(*d["tokenized"])

                    
                    # Build dictionary of words-to-idx-of-embeddings-of-where-they-appear
                    words2embeddingidx = { w:[] for w in all_words_t2 }
                    for (idx_row, row), embeddings in zip(d.iterrows(), embeddings1):
                        for idx_sentence, tokenized_sentence in enumerate(row['tokenized_sentences']):
                            for word in tokenized_sentence:
                                words2embeddingidx[word].append( (idx_row, idx_sentence)) 

                    # Use that dictionary to get the sentence embedding for each context where the word has been used (per time, per speaker, etc)
                    words2embeddings2 = { word:[ embeddings2.loc[idx_row][idx_sentence] for (idx_row,idx_sentence) in idxs]
                                                                                         for word,idxs in words2embeddingidx.items() }

                    # Take the centroid of those sentence embeddings, make a dictionary of words and their centroids
                    words2centroids2  = { word:np.mean(embeddings,axis=0) for word,embeddings in words2embeddings2.items() }

                    # Since they are pre-trained, model1 and model2 are already aligned. We just need to make sure they share the same vocabulary.
                    common_vocab = all_words_t1.intersection(all_words_t2)
                    self.logger.info('MODELLING - WHOLE - ALIGNMENT COMPLETE')
                    if len(common_vocab) == 0:
                        self.logger.error("MODELLING - WHOLE - NO COMMON VOCAB LEFT OVER")

                    # Using the KeyedVectors (https://radimrehurek.com/gensim_3.8.3/models/word2vec.html), build word2vec models from those vectors
                    from gensim.models import KeyedVectors
                    words    = common_vocab

                    vectors1 = np.array([ words2centroids1[word] for word in words ])
                    model1   = KeyedVectors(vectors1.shape[1])
                    model1.add_vectors(words, vectors1)

                    vectors2 = np.array([ words2centroids2[word] for word in words ])
                    model2   = KeyedVectors(vectors2.shape[1])
                    model2.add_vectors(words, vectors2)        

                    self.model1.save(savepath_t1)
                    self.logger.info('MODELLING - WHOLE - MODEL 1 SAVED')
                    self.model2.save(savepath_t2)
                    self.logger.info('MODELLING - WHOLE - MODEL 2 SAVED')


            if self.model_type in ['speaker', 'speaker_plus']:

                """
                With the speaker model, we train word embeddings for each MP and split into two time groups as well, t1 and t2.
                """

                self.split_speeches_by_mp = self.split_speeches_df(by='speaker')
                self.speaker_saved_models = []
                new = False
                count = 0
                skipped = 0
                vocab_skipped = 0
                for row in self.split_speeches_by_mp.itertuples():
                    model_savepath = os.path.join(outdir, f'speaker_{row.df_name}.model.sbert')

                    # if the file exists and we are not overwriting, record that is is there.
                    if (os.path.isfile(model_savepath) and not overwrite):
                        self.logger.info(f'{os.path.split(model_savepath)[-1]} found and not overwriting...')
                        self.speaker_saved_models.append(model_savepath)

                    # if the file does not exist and we are not checking if it should, just continue
                    elif (not os.path.isfile(model_savepath)) and skip_model_check:
                        self.logger.info(f'{os.path.split(model_savepath)[-1]} not found but skipping model check...')
                        continue

                    # otherwise, generate model
                    else:
                        try:
                            count += 1
                            model = gensim.models.Word2Vec(
                                row.tokens,
                                min_count=1,
                                vector_size=300,
                                window = 5,
                                sg = 1
                            )
                            vocab_of_interest = set(self.change)
                            req_size = len(vocab_of_interest)
                            overlap = len(vocab_of_interest.intersection(set(model.wv.index_to_key)))/req_size
                            if overlap<overlap_req:
                                vocab_skipped += 1
                                self.logger.info(f'MODELLING - Skipped {row.df_name} due to not enough overlap with words of interest. Overlap: {overlap:.2f}')
                                continue
                            # Skip if below minimum size
                            if len(model.wv.index_to_key) < min_vocab_size:
                                skipped += 1
                                self.logger.info(f'MODELLING - SKIPPED {row.df_name} due to insufficient vocab size. Vocab size: {len(model.wv.index_to_key)}')
                                continue
                            if not new:
                                new = True
                                self.logger.info('New set to True')
                            model.save(model_savepath)
                            if count % 100 == 0:
                                self.logger.info(f'MODELLING - {count}/{len(self.split_speeches_by_mp)} = {100*count/len(self.split_speeches_by_mp):.2f}% complete')
                            # self.logger.info(f'MODELLING - SPEAKER - Saved model to {model_savepath}.')
                            self.speaker_saved_models.append(model_savepath)
                        except Exception as e:
                            self.logger.error(e)

                self.logger.info(f"num speaker models: {len(self.speaker_saved_models)}")
                if count > 0:
                    self.logger.info(f"MODELLING - SPEAKER - {skipped} out of {count}  models skipped due to vocab size")
                    self.logger.info(f"MODELLING - SPEAKER - {vocab_skipped} out of {count} models skipped due to insufficient overlap with vocab of interest")
                    self.logger.info(f"Total skipped: {vocab_skipped+skipped} out of {count} = {100*(vocab_skipped+skipped)/count:.2f}%")
                else:
                    self.logger.info(f"No models made or skipped")

                if new and align:
                    self.logger.info('MODELLING - SPEAKER - New models detected. Loading back in and running alignment')
                    for ind, model_path in enumerate(self.speaker_saved_models[:-1]):

                        model_current = gensim.models.Word2Vec.load(model_path)
                        check = np.array(model_current.wv[model_current.wv.index_to_key[0]])
                        model_next    = gensim.models.Word2Vec.load(self.speaker_saved_models[ind+1])

                        _ = self._smart_procrustes_align_gensim(model_current, model_next)

                        current_common_vocab_size = len(set(model_current.wv.index_to_key).intersection(set(model_next.wv.index_to_key)))
                        self.logger.info(f"MODELLING - SPEAKER - ALIGNMENT - CURRENT COMMON VOCAB IS {current_common_vocab_size} after alignment at index {ind}, model: {model_path}")

                        if np.sum(check-model_current.wv[model_current.wv.index_to_key[0]])>0:
                            self.logger.warning('MODELLING - SPEAKER - PLEASE CHECK ALIGNMENT PROCEDURE')
                            return None

                        model_current.save(model_path)
                        model_next.save(self.speaker_saved_models[ind+1])
                    self.logger.info('MODELLING - SPEAKER - ALIGNMENT COMPLETE')
                    if current_common_vocab_size == 0:
                        self.logger.error("MODELLING - SPEAKER - NO COMMON VOCAB LEFT OVER")
                else:
                    self.logger.info('MODELLING - SPEAKER - NO NEW MODELS GENERATED -> NO ALIGNMENT NECESSARY')

            if self.model_type == 'retrofit':
                # create aligned models for retrofit.

                self.logger.info('MODELLING - RETROFIT')
                self.retrofit_model_paths = []
                new = False
                count = 0
                skipped = 0
                vocab_skipped = 0
                for row in self.retrofit_prep_df.itertuples(): 

                    savepath = os.path.join(outdir, f"{row.df_name}.model.sbert")
                    if (os.path.isfile(savepath) and overwrite) or (not os.path.isfile(savepath) and not skip_model_check):
                        model = gensim.models.Word2Vec(
                            row.tokens,
                            min_count=1,
                            vector_size=300,
                            window = 5,
                            sg = 1
                        )

                        count += 1
                        # Skip if not containing correct vocab
                        vocab_of_interest = set(self.change)
                        req_size = len(vocab_of_interest)
                        overlap = len(vocab_of_interest.intersection(set(model.wv.index_to_key)))/req_size
                        if overlap<overlap_req:
                            vocab_skipped += 1
                            self.logger.debug(f'Modelling: Skipped {row.df_name} due to not enough overlap with words of interest. Overlap: {overlap:.2f}')
                            continue
                        # Skip if below minimum size
                        if len(model.wv.index_to_key) < min_vocab_size:
                            skipped += 1
                            self.logger.debug(f'MODELLING - SKIPPED {row.df_name} due to insufficient vocab size. Vocab size: {len(model.wv.index_to_key)}')
                            continue

                        if not new:
                            new = True
                            self.logger.info('New set to True')

                        if count % 100 == 0:
                            self.logger.info(f'MODELLING - {count}/{len(self.retrofit_prep_df)} = {100*count/len(self.retrofit_prep_df):.2f}% complete')

                        # N.B. only append savepath if retrofit model satisfies criterion.
                        self.retrofit_model_paths.append(savepath)

                        # Previous: Also saving model in a dict and exporting
                        # Updated 22/10/22: save as you go along for RAM reasons. Also just better

                        model.save(savepath)

                        # dictOfModels[dframe] = model
                        #model.save(os.path.join(models_folder, modelName))
                    elif os.path.isfile(savepath):
                        self.retrofit_model_paths.append(savepath)
                        count += 1

                self.logger.info(f"MODELLING - RETROFIT - {skipped} out of {count}  models skipped due to vocab size")
                self.logger.info(f"MODELLING - RETROFIT - {vocab_skipped} out of {count} models skipped due to insufficient overlap with vocab of interest")
                self.logger.info(f"Total skipped: {vocab_skipped+skipped} out of {count} = {100*(vocab_skipped+skipped)/count:.2f}%")

                if new and align:
                    self.logger.info('MODELLING - RETROFIT - New models detected. Loading back in and running alignment')
                    for ind, model_path in enumerate(self.retrofit_model_paths[:-1]):

                        model_current = gensim.models.Word2Vec.load(model_path)
                        check = np.array(model_current.wv[model_current.wv.index_to_key[0]])
                        model_next    = gensim.models.Word2Vec.load(self.retrofit_model_paths[ind+1])

                        _ = self._smart_procrustes_align_gensim(model_current, model_next)

                        current_common_vocab_size = len(set(model_current.wv.index_to_key).intersection(set(model_next.wv.index_to_key)))
                        self.logger.debug(f"MODELLING - ALIGNMENT - CURRENT COMMON VOCAB IS {current_common_vocab_size} after alignment at index {ind}, model: {model_path}")

                        if np.sum(check-model_current.wv[model_current.wv.index_to_key[0]])>0:
                            self.logger.warning('MODELLING - RETROFIT - PLEASE CHECK ALIGNMENT PROCEDURE')
                            return None

                        model_current.save(model_path)
                        model_next.save(self.retrofit_model_paths[ind+1])
                    self.logger.info('MODELLING - RETROFIT - ALIGNMENT COMPLETE')
                    if current_common_vocab_size == 0:
                        self.logger.error("MODELLING - RETROFIT - NO COMMON VOCAB LEFT OVER")
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
            word_count: total count across t1 or t2
        """
        if dictKeyVector:
            modelsSum = np.zeros(np.array(dictKeyVector[list(dictKeyVector.keys())[0]]).shape[0])
        else:
            modelsSum = np.zeros(self.dictOfModels['t1'][0].layer1_size)

        # count for seeing how many to divide by
        count = 0
        word_count = 0
        if dictKeyVector:
            for key, value in dictKeyVector.items():
                if w in key and time in key:
                    modelsSum = np.add(modelsSum, value)
                    count += 1
        else:
            for model in self.dictOfModels[time]:
                try:
                    modelsSum = np.add(modelsSum, model.wv[w])
                    word_count += model.wv.get_vecattr(w, "count")
                    count += 1
                except KeyError:
                    continue

        if count == 0:
            if self.verbosity > 0:
                print(f'Word "{w}" not found')
            return modelsSum, word_count
        else:
            avgEmbedding = np.divide(modelsSum, count)
            return avgEmbedding, word_count

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

    def woi(self):

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

            self.words_of_interest = self.cosine_similarity_df[self.cosine_similarity_df['Word'].isin(self.change+self.no_change)].copy()

            self.words_of_interest.loc[self.words_of_interest['Word'].isin(self.change), 'semanticDifference'] = 'change'
            self.words_of_interest.loc[self.words_of_interest['Word'].isin(self.no_change), 'semanticDifference'] = 'no_change'

            self.change_cossim = self.words_of_interest.loc[self.words_of_interest['semanticDifference'] == 'change', 'Cosine_similarity'] 
            self.no_change_cossim = self.words_of_interest.loc[self.words_of_interest['semanticDifference'] == 'no_change', 'Cosine_similarity'] 

            return self.words_of_interest, self.change_cossim, self.no_change_cossim

        elif self.model_type in ['speaker', 'speaker_plus']:

            self.words_of_interest = self.cosine_similarity_df[self.cosine_similarity_df['Word'].isin(self.change+self.no_change)].copy()

            self.words_of_interest.loc[:,'FrequencyRatio'] = self.words_of_interest['Frequency_t1']/self.words_of_interest['Frequency_t2']
            self.words_of_interest.loc[:,'TotalFrequency'] = self.words_of_interest['Frequency_t1'] + self.words_of_interest['Frequency_t2']

            self.words_of_interest.loc[self.words_of_interest['Word'].isin(self.change), 'semanticDifference'] = 'change'
            self.words_of_interest.loc[self.words_of_interest['Word'].isin(self.no_change), 'semanticDifference'] = 'no_change'

            return self.words_of_interest

        elif self.model_type in ['retrofit', 'retro']:

            self.words_of_interest = self.change + self.no_change

    def postprocess(self, model_output_dir, workers=10, retrofit_factor=None, overwrite=False)->None:
        self.logger.info("POSTPROCESS: BEGIN")
        if self.model_type == 'speaker':
            self.process_speaker(model_output_dir, overwrite=overwrite)
        elif self.model_type == 'speaker_plus':
            self.process_speaker_plus()

        self.woi()

        if self.model_type in ['retrofit', 'retro']:
            self.retrofit_main_create_synonyms(factor = retrofit_factor, overwrite=overwrite)
            self.retrofit_create_input_vectors(workers = workers, overwrite=overwrite)
            self.retrofit_output_vec(model_output_dir = model_output_dir, overwrite=overwrite)
            self.retrofit_post_process(self.change, self.no_change, model_output_dir)

    def process_speaker_plus(self):

        # 2022-12-06: Speaker plus: take cosine similarity of the same word in t1 and word in t2, with no averaging
        # self.valid_split_speeches_by_mp = []
        # self.dictOfModels = {
        #     't1': [],
        #     't2': []
        # }
        # total_words = set()
        # for model_savepath in self.speaker_saved_models:
        #     model = gensim.models.Word2Vec.load(model_savepath)
        #     # if len(model.wv.index_to_key) < min_vocab_size:
        #         # continue
        #     if 't1' in model_savepath:
        #         self.dictOfModels['t1'].append(model)
        #     else:
        #         self.dictOfModels['t2'].append(model)
        #     # else:
        #     self.valid_split_speeches_by_mp.append(model_savepath)
        #     total_words.update(model.wv.index_to_key)
        # self.logger.info(f'POSTPROCESS - SPEAKER - Total words: {len(total_words)}')

        self.pairs_of_models = []
        self.dictOfModels = {
            't1': [],
            't2': []
        }
        total_words = set()
        for model_savepath in self.speaker_saved_models:
            if 't1' in model_savepath:
                model1 = gensim.models.Word2Vec.load(model_savepath)
                t2_string = model_savepath.replace('t1','t2')
                if t2_string in self.speaker_saved_models:
                    model2 = gensim.models.Word2Vec.load(t2_string)
                    self.pairs_of_models.append((model1,model2))
                    total_words.update(model1.wv.index_to_key)
                    total_words.update(model2.wv.index_to_key)
                    self.dictOfModels['t1'].append(model1)
                    self.dictOfModels['t2'].append(model2)
        self.logger.info(f'Pairs: {len(self.pairs_of_models)}')

        word_sims = defaultdict(list)
        for index, word in enumerate(total_words): 
            if index % 100 == 0:
                self.logger.info(f'Processed {index} of {len(total_words)} = {100*index/len(total_words):.2f}%')
            for m1, m2 in self.pairs_of_models:
                if word in m1.wv.index_to_key and word in m2.wv.index_to_key:
                    word_sims[word].append(self.cosine_similarity(m1.wv[word],m2.wv[word]))
        #     for t1_model in self.dictOfModels['t1']:
        #         # get string for t2 model
        #         t2_model_str = t1_
        #         # for t2_model in self.dictOfModels['t2']:
        #             if word in t1_model.wv.index_to_key and word in t2_model.wv.index_to_key:
        #                 word_sims[word].append(self.cosine_similarity(t1_model.wv[word], t2_model.wv[word]))
        word_vals = []
        for word, values in word_sims.items():
            word_vals.append({
                'Word': word,
                'mean_cossim': np.mean(values),
                'var_cossim': np.var(values),
                'Frequency_t1': self.computeAvgVec(word, time='t1')[1],
                'Frequency_t2': self.computeAvgVec(word, time='t2')[1]
            })
        self.cosine_similarity_df = pd.DataFrame.from_records(word_vals)

    def process_speaker(self,
        model_output_dir,
        min_vocab_size=10000,
        overwrite=False
    ):
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
        self.logger.info(f'POSTPROCESS - SPEAKER - Total words: {len(total_words)}')

        # print, for informational purposes, the number of models that are over the threshold
        self.logger.info(f'POSTPROCESS - SPEAKER - Number of valid models: {len(self.valid_split_speeches_by_mp)}')

        avg_vec_savepath_t1 = os.path.join(model_output_dir,'average_vecs_t1.bin')
        avg_vec_savepath_t2 = os.path.join(model_output_dir,'average_vecs_t2.bin')

        # if ((os.path.isfile(avg_vec_savepath_t1) or os.path.isfile(avg_vec_savepath_t2)) and overwrite) or not (os.path.isfile(avg_vec_savepath_t1) and os.path.isfile(avg_vec_savepath_t2)):
        average_vecs = {
            't1': {},
            't2': {}
        }
        self.cosine_similarity_df = pd.DataFrame(columns = (
            'Word',
            'Frequency_t1',
            'Frequency_t2',
            'Cosine_similarity'
        ))
        self.logger.info(f'POSTPROCESS - SPEAKER - CALCULATING AVERAGE VECTORS')
        for word in total_words:
            if self.verbosity > 0:
                self.logger.info(f'POSTPROCESS - SPEAKER - getting average vector for {word}')
            avgVecT1, freq_t1 = self.computeAvgVec(word, time='t1')
            avgVecT2, freq_t2 = self.computeAvgVec(word, time='t2')

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
                    "Frequency_t1": freq_t1,
                    "Frequency_t2": freq_t2,
                    "Cosine_similarity": cosSimilarity
                }

                self.cosine_similarity_df = pd.concat([self.cosine_similarity_df, pd.DataFrame([insert_row])], axis=0)

        self._save_word2vec_format(
            fname = avg_vec_savepath_t1,
            vocab = average_vecs['t1'],
            vector_size = average_vecs['t1'][list(average_vecs['t1'].keys())[0]].shape[0]
        )
        self.logger.info(f'POSTPROCESS - SPEAKER - Average vectors for t1 saved to {avg_vec_savepath_t1}')
        self._save_word2vec_format(
            fname = avg_vec_savepath_t2,
            vocab = average_vecs['t2'],
            vector_size = average_vecs['t2'][list(average_vecs['t2'].keys())[0]].shape[0]
        )
        self.logger.info(f'POSTPROCESS - SPEAKER - Average vectors for t2 saved to {avg_vec_savepath_t2}')

        self.model1 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t1, binary=True)
        self.model2 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t2, binary=True)

    def process_BERT_speaker(self,
        model_output_dir,
        min_vocab_size=10000,
        overwrite=False
    ):
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
        self.logger.info(f'POSTPROCESS - SPEAKER - Total words: {len(total_words)}')

        # print, for informational purposes, the number of models that are over the threshold
        self.logger.info(f'POSTPROCESS - SPEAKER - Number of valid models: {len(self.valid_split_speeches_by_mp)}')

        avg_vec_savepath_t1 = os.path.join(model_output_dir,'average_vecs_t1.bin')
        avg_vec_savepath_t2 = os.path.join(model_output_dir,'average_vecs_t2.bin')

        # if ((os.path.isfile(avg_vec_savepath_t1) or os.path.isfile(avg_vec_savepath_t2)) and overwrite) or not (os.path.isfile(avg_vec_savepath_t1) and os.path.isfile(avg_vec_savepath_t2)):
        average_vecs = {
            't1': {},
            't2': {}
        }
        self.cosine_similarity_df = pd.DataFrame(columns = (
            'Word',
            'Frequency_t1',
            'Frequency_t2',
            'Cosine_similarity'
        ))
        self.logger.info(f'POSTPROCESS - SPEAKER - CALCULATING AVERAGE VECTORS')
        for word in total_words:
            if self.verbosity > 0:
                self.logger.info(f'POSTPROCESS - SPEAKER - getting average vector for {word}')
            avgVecT1, freq_t1 = self.computeAvgVec(word, time='t1')
            avgVecT2, freq_t2 = self.computeAvgVec(word, time='t2')

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
                    "Frequency_t1": freq_t1,
                    "Frequency_t2": freq_t2,
                    "Cosine_similarity": cosSimilarity
                }

                self.cosine_similarity_df = pd.concat([self.cosine_similarity_df, pd.DataFrame([insert_row])], axis=0)

        self._save_word2vec_format(
            fname = avg_vec_savepath_t1,
            vocab = average_vecs['t1'],
            vector_size = average_vecs['t1'][list(average_vecs['t1'].keys())[0]].shape[0]
        )
        self.logger.info(f'POSTPROCESS - SPEAKER - Average vectors for t1 saved to {avg_vec_savepath_t1}')
        self._save_word2vec_format(
            fname = avg_vec_savepath_t2,
            vocab = average_vecs['t2'],
            vector_size = average_vecs['t2'][list(average_vecs['t2'].keys())[0]].shape[0]
        )
        self.logger.info(f'POSTPROCESS - SPEAKER - Average vectors for t2 saved to {avg_vec_savepath_t2}')

        self.model1 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t1, binary=True)
        self.model2 = gensim.models.KeyedVectors.load_word2vec_format(avg_vec_savepath_t2, binary=True)

    def retrofit_create_synonyms(self, words):
        """Function to create the synonyms from an input dataframe (retrofit_prep)

        Args:
            data (_type_): _description_
            word (_type_): _description_
            factor (_type_): _description_

        Returns:
            _type_: _description_
        """

        self.logger.info(f'RETROFIT - CREATE SYNONYMS FOR WORDS OF INTEREST')

        potential_factors = ['party', 'time', 'debate']

        # self.logger.info(f"RETROFIT FACTOR: {self.retrofit_factor}")
        #TODO: ENSURE DATA SLICING IS CORRECT
        parties = list(self.data.party.unique())
        parties = [i for i in parties if isinstance(i, str)]
        # To fix party names like 'Scottish National Party by inserting hyphens between
        # parties = [i.replace(' ','_') for i in parties]

        # collect debate id list
        # debate_id_set = set()
        # for debate_id_list in self.data['debate_id']:
            # debate_id_set.add(str(debate_id))
        debate_id_set = set(self.data['debate_id'].astype(str).unique())
        assert len(debate_id_set) > 0
        debate_id_list = list(debate_id_set)

        # set times
        times = ['t1', 't2']

        identifier_dict = {
            'party': parties,
            'time': times,
            'debate': debate_id_list,
        }
        identifier_factors = [list(set(words))]
        for potential_factor in potential_factors:
            if potential_factor in self.retrofit_factor:
                identifier_factors.append(identifier_dict[potential_factor])
                self.logger.debug(f'added {potential_factor}')

        # self.logger.debug('Generate identifiers')
        # raw_identifiers = list(product(*identifier_factors))
        # self.logger.debug(f'{len(raw_identifiers)} product identifiers, e.g.: {raw_identifiers[:10]}')
        # identifiers = [syn_identifier(i) for i in raw_identifiers]
        # identifiers = [syn_identifier(*i) for i in raw_identifiers]
        # self.logger.debug(f'{identifiers[0]}')
        # self.logger.debug(f"Exapmle syn_identifier: {identifiers[0].stringify()}")
        # self.logger.info('RETROFIT - CREATE SYNONYMS - IDENTIFIERS GENERATED')

        # initiate dictionary to save output for
        # dictOfSynonyms={}

        # # Iterate parties & create synonyms where more than one record for a party
        # temp = True
        # # self.logger.info(f'{len(identifiers)} to scan.')
        # count = 0
        # for ind, identifier in enumerate(identifiers):
        #     # self.logger.debug(f'Running identifier {identifier.stringify()}')

        #     # if ind % 100000 == 0:
        #         # self.logger.info(f'Processed {ind} of {len(identifiers)} = {100*ind/len(identifiers):.2f}%')

        #     selected_df = self.data.copy()
        #     for potential_factor in potential_factors:
        #         if potential_factor in self.retrofit_factor: 
        #             if potential_factor == 'party':
        #                 selected_df = selected_df[selected_df['party']==identifier.party]
        #                 # self.logger.debug(f'Party factor detected for selected df')
        #             if potential_factor == 'time':
        #                 selected_df = selected_df[selected_df['time']==identifier.time]
        #                 # self.logger.debug(f'Time factor detected for selected df')
        #             if potential_factor == 'debate':
        #                 # selected_df = selected_df[selected_df['debate_id'].apply(lambda x: int(identifier.debate) in x)]
        #                 selected_df = selected_df[selected_df['debate_id'] == int(identifier.debate)]
        #     temp=False

        #     if 'debate' not in self.retrofit_factor:
        #         # if no debate, there will be lots of duplicates
        #         for potential_factor in  potential_factors:
        #             selected_df = selected_df.groupby(potential_factor).first()
        #     identifier_synonyms=[]

        #     # speaker_ids=list(selected_df['speaker'].unique())

        #     # temp = True
        #     for row in selected_df.itertuples():
        #         if 'debate' in self.retrofit_factor:
        #             tokens = set(row.tokenized)
        #             if word in tokens:
        #                 syn = synonym_item(
        #                     word = word,
        #                     time = identifier.time,
        #                     speaker = row.speaker,
        #                     party  = row.party
        #                 )
        #                 identifier_synonyms.append(syn)
        #                 count += 1
        #         else:
        #             syn = synonym_item(
        #                 word = word,
        #                 time = identifier.time,
        #                 speaker = row.speaker,
        #                 party  = row.party
        #             )
        #             identifier_synonyms.append(syn)
        #             count += 1

        #     dictOfSynonyms[identifier.stringify()]=identifier_synonyms
        # self.logger.info(f'{count} string saved')

        # 2022-12-05 New Logic: Can we iterate over the data table only once?
        # Save output to a dictionary that has identifiers as the keys, and only add in when they match the identifiers.
        # stringified_identifiers = [i.stringify() for i in identifiers]
        # output_dict = {k: [] for k in stringified_identifiers}
        output_dict = defaultdict(list)
        length = len(self.data)
        for row in self.data.itertuples():
            if row.Index % 10000 == 0:
                self.logger.info(f'{row.Index} rows processed = {100*row.Index/length:.2f}%')
            overlap = set(words).intersection(row.token_set)
            if len(overlap) == 0:
                # can be no overlap in terms of interest, if so, continue.
                continue
            else:
                # loop over words that are contained in this debate
                for word in overlap:
                    t = None
                    d = None
                    if 'time' in self.retrofit_factor:
                        t = row.time
                    if 'debate' in self.retrofit_factor:
                        d = str(row.debate_id)
                    # create identifier and match against output_dit
                    temp_identifier = syn_identifier(
                        word = word,
                        party = row.party,
                        time = t,
                        debate = d
                    ).stringify()
                    # if temp_identifier in output_dict:
                    syn_item = synonym_item(
                        word = word,
                        time = t,
                        speaker = row.speaker,
                        party = row.party,
                    )
                    output_dict[temp_identifier].append(syn_item)

        self.logger.info(f'Number of synonym keys: {len(output_dict)}')

        # self.logger.info('Clean output to remove empty lists')


        return output_dict

    def retrofit_main_create_synonyms(self, factor = None, overwrite=False):

        # Sanity check
        assert self.retrofit_prep_df is not None

        # Save factor
        self.retrofit_factor = factor

        # Set paths for pickle and text paths of lexicon
        self.synPicklePath = os.path.join(self.retrofit_outdir, f'synonyms_{self.parliament_name}_{factor}.pkl')
        self.synTextPath = os.path.join(self.retrofit_outdir, f'synonyms_{self.parliament_name}_{factor}.txt')

        self.logger.info(f'RETROFIT - MAIN CREATING SYNONYMS')
        if ((os.path.isfile(self.synPicklePath) and os.path.isfile(self.synTextPath)) and overwrite) or not (os.path.isfile(self.synPicklePath) and os.path.isfile(self.synTextPath)):

            # allSynonyms=[]
            # for word in self.words_of_interest:
                # synonymsPerWord = self.retrofit_create_synonyms(self.retrofit_prep_df,word,self.retrofit_factor)
                # print(len(synonyms)) #Verify length of synonyms
                # allSynonyms.append(synonymsPerWord)

            # with ProcessPoolExecutor(max_workers=10) as executor:
                # results = executor.map(self.retrofit_create_synonyms, self.words_of_interest)
            total_dict = self.retrofit_create_synonyms(self.words_of_interest)

            # 2022-12-01: Now each synonymsPerWord is a dictionary
            # total_dict = {k:v for i in results for k,v in i.items()}

            #Here it is 84 , which is sum of combinations made 
            #for the three parties (13,3,3)=> no. of combinations is (78,3,3), 78+3+3= 84, hence verified. 

            # We're capturing synonyms of all words of interest regardless of whether they're part of the models' vocab
            # Since the same synonyms-dictionary can be used for other models
            #print(len(words_of_interest),len(allSynonyms))

            # allSynonyms = [tup for lst in allSynonyms for tup in lst]
            #print(len(allSynonyms)) 
            # For party factor alone =>Length should be 187*84=15708 OR len(words_of_interest)*len(mp-in-same-party pairs)
            # For party-time factor => Length should be 187*42=7854 OR len(w_of_int)*len(mp-in-same-party-same-time pairs)

            # Writing synonym files 
            # Change name for the pkl and txt files as per synonym-making factor, e.g. synonyms-party-time, etc

            with open(self.synPicklePath, 'wb') as f:
                pickle.dump(total_dict, f)

            with open(self.synTextPath,'w') as f:
                for _, v in total_dict.items():
                    for syn_str in v:
                        f.write(syn_str.stringify())
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
            for word in self.words_of_interest:
                syn = synonym_item(
                    word = word,
                    time=row.time,
                    speaker=row.speaker,
                    party=row.party
                )
                if word in model.wv.index_to_key:
                    result[index_count, :] = model.wv[word]
                    index_to_key.append(syn.stringify())
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
        """This function collects the vectors required for retrofitting from the speaker Word2Vec models. Some superfluous vectors will be collected, but there will be no reference errors when it comes to retrofitting.

        Args:
            workers (_type_, optional): _description_. Defaults to None.
            overwrite (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        self.logger.info('RETROFIT - CREATE INPUT VECTORS')

        # define output filenames
        self.vectorFileName = os.path.join(self.retrofit_outdir,f'vectors_{self.retrofit_factor}.hdf5')
        self.vectorIndexFileName = os.path.join(self.retrofit_outdir,f'vector_index_to_key_{self.parliament_name}_{self.retrofit_factor}.pkl')

        # sanity check that we do have retrofit savepaths readily accesible
        assert len(self.retrofit_model_paths) > 0

        first=True

        if (os.path.isfile(self.vectorFileName) and overwrite) or not os.path.isfile(self.vectorFileName):

            # Load in detected sysnonyms for each word
            with open(self.synPicklePath, 'rb') as f:
                synonyms = pickle.load(f)

            # split tuples of synonyms up
            # firstSyns = [tup[0] for tup in synonyms]
            # secondSyns = [tup[1] for tup in synonyms]
            # synonymsList = firstSyns+secondSyns
            self.total_syn_list = [v for v in synonyms.values()]
            self.total_syn_list = [item for sublist in self.total_syn_list for item in sublist]
            # uniqueSynonymsList = set(total_syn_list)

            syn_df = pd.DataFrame(columns = ['full_model_path','modelKey', 'time', 'speaker', 'party', 'debate', 'debate_id'])
            syn_df['full_model_path'] = self.retrofit_model_paths
            syn_df['modelKey'] = [os.path.split(i)[-1] for i in self.retrofit_model_paths]
            syn_df['time'] = syn_df['modelKey'].apply(lambda x: x.split('df_')[1].split('_')[0])
            syn_df['speaker'] = syn_df['modelKey'].apply(lambda x: re.split('t[12]_', x.split('df_')[1])[1])
            syn_df['speaker'] = syn_df['speaker'].apply(lambda x: x.replace(' ','_'))
            syn_df['speaker'] = syn_df['speaker'].apply(lambda x: x.split('.model')[0])
            # syn_df['party'] = syn_df['speaker'].apply(lambda x: self.retrofit_prep_df[self.retrofit_prep_df['speaker'] == x]['party'].iat[0])
            # self.logger.info(syn_df['speaker'].unique())
            # self.logger.info(self.data['speaker'].unique())
            syn_df['party'] = syn_df['speaker'].apply(lambda x: self.data[self.data['speaker'] == x]['party'].iat[0])
            # syn_df['party'] = 

            syn_df = syn_df[syn_df['full_model_path'].apply(lambda x: os.path.isfile(x))]

            # syn_df['debate'] = syn_df.apply(lambda x: self.retrofit_prep_df[(self.retrofit_prep_df['speaker'] == x.speaker) & (self.retrofit_prep_df['df_name'].isin(x.time))]['debate'].iat[0], axis=1)
            # syn_df['debate_id'] = syn_df.apply(lambda x: self.retrofit_prep_df[(self.retrofit_prep_df['speaker'] == x.speaker) & (self.retrofit_prep_df['df_name'].isin(x.time))]['debate_id'].iat[0], axis=1)

            # mpNamePartyInfo is meant to have stuff like '-Con' for Conservatives
            # mpNames = []

            self.logger.info(f'RETROFIT - CHECKING WHICH SPEAKERS TO KEEP WITH GENEREATED SYNONYMS...')
            # iterate over each speaker model in syn_df. Then search the unique synonym list to see if they have words in there. If so, save the associated party info.
            speakers_in_syn_list = set([i.speaker for i in self.total_syn_list])
            syn_df = syn_df[syn_df['speaker'].isin(speakers_in_syn_list)]
            syn_df = syn_df.reset_index()
            self.logger.info(f'RETROFIT - DONE CHECKING WHICH SPEAKERS TO KEEP WITH GENEREATED SYNONYMS.')

            # retrieve required vector size from a file
            temp_model = gensim.models.Word2Vec.load(self.retrofit_model_paths[0])
            temp_vec = temp_model.wv[temp_model.wv.index_to_key[0]]
            self.vector_size = temp_vec.shape[0]

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

            # sanity check: all the keys in the synonym list should be in the retrieved vectors.
            # self.logger.info('Performing sanity check')
            # set_index_to_key = set(index_to_key)
            # all_stringified = set([i.stringify() for i in self.total_syn_list])
            # if all_stringified.issubset(set_index_to_key):
            #     self.logger.info('Sanity check PASSED! :)')
            # else:
            #     syn_items_with_no_key = all_stringified - all_stringified.intersection(set_index_to_key)
            #     self.logger.warning(f'{len(syn_items_with_no_key)}')
            #     self.logger.warning(syn_items_with_no_key)

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
        self.logger.info('Retrofit: Generate outfile')
        self.retrofit_outfile = os.path.join(model_output_dir,f'retrofit_out_{self.retrofit_factor}.txt')
        if (os.path.isfile(self.retrofit_outfile) and overwrite) or not os.path.isfile(self.retrofit_outfile):
            wordVecs = self.retrofit_read_word_vecs_hdf5()
            lexicon = retrofit.read_lexicon(self.synTextPath)
            numIter = int(10)

            ''' Enrich the word vectors using ppdb and print the enriched vectors '''
            retrofit.print_word_vecs(retrofit.retrofit(wordVecs, lexicon, numIter), self.retrofit_outfile)
        else:
            self.logger.info(f'Retrofit: Retrofit file already exists at {self.retrofit_outfile}')

    def retrofit_post_process(self, change, no_change, model_output_dir):
        self.logger.info('Retrofit: Post Processing')

        # retrofit outfile is of structure: synKey vector <- for each line
        with open(self.retrofit_outfile) as f:
            vecs = f.readlines()
            vecs = [vec.replace('\n', '')for vec in vecs]
            # vecs=[]
            # vec=''

            # while True:
            #     line = f.readline()
            #     if not line:
            #         break
            #     # check the line start is alphanumeric
            #     if (str(list(line)[0]).isalpha()):
            #         vec=vec.strip()
            #         if(vec!=''):
            #             vecs.append(vec)
            #         vec = line
            #     else:
            #         vec += line
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
                dictKeyVector[synKey]=np.array(vec)
                # npVec = np.array(dictKeyVector[synKey])
        self.logger.info(f'Count of vectors with fewer dimensions that we will not consider: {count}')
        dfRetrofitted = pd.DataFrame({'vectorKey':list(dictKeyVector.keys()), 'vectors':list(dictKeyVector.values())})
        self.logger.debug(f'dfRetrofitted dimensions:{dfRetrofitted.shape}')

        # Filtering down words of interest as per those present in our vectors 
        # We're amending the computeAvgVec function accordingly
        # As it calculated based on processing from models, and here we're only taking vectors. Hence this check here too.

        vectorKeys = list(dfRetrofitted['vectorKey'])
        vectorKeys = [synonym_item.from_string(i) for i in vectorKeys]
        # Extracting words from vectors keys
        words_from_syn_keys= list(set([key.word for key in vectorKeys]))
        assert all(['$' not in i for i in words_from_syn_keys])
        assert all(['-' not in i for i in words_from_syn_keys ])
        # print(words_of_interest, len(words_of_interest))

        self.cosine_similarity_df = pd.DataFrame(columns = (
            'Word',
            'Frequency_t1',
            'Frequency_t2',
            'Cosine_similarity'
        ))

        # NOW WE ONLY HAVE THOSE WORDS HERE WHICH ARE PRESENT IN THE VECTORS.
        # t1Keys = [t for t in list(dictKeyVector.keys()) if 't1' in t]
        # t2Keys = [t for t in list(dictKeyVector.keys()) if 't2' in t]
        sims= []

        # Compute average of word in T1 and in T2 and store average vectors and cosine difference
        avg_vec_dict_t1 = {}
        avg_vec_dict_t2 = {}
        for word in words_from_syn_keys:

            #Provide a list of keys to average computation model for it to
            # #compute average vector amongst these models
            # wordT1Keys = [k for k in t1Keys if k.split('-')[0]==word]
            # wordT2Keys = [k for k in t2Keys if k.split('-')[0]==word]

            #Since here the key itself contains the word we're not simply sending T1 keys but sending word-wise key
            avgVecT1, _ = self.computeAvgVec(word, time = 't1', dictKeyVector = dictKeyVector)
            avgVecT2, _ = self.computeAvgVec(word, time = 't2', dictKeyVector = dictKeyVector)
            avg_vec_dict_t1[word] = avgVecT1
            avg_vec_dict_t2[word] = avgVecT2

            if(avgVecT1.shape == avgVecT2.shape):
                # Cos similarity between averages
                cosSimilarity = self.cosine_similarity(avgVecT1, avgVecT2)
                sims.append(cosSimilarity)
            else:
                self.logger.info('Word not found')
        word_count_dict_t1 = self._get_retrofit_word_counts(words_from_syn_keys, time='t1')
        word_count_dict_t2 = self._get_retrofit_word_counts(words_from_syn_keys, time='t2')
        self.cosine_similarity_df['Word']=words_from_syn_keys
        self.cosine_similarity_df['Cosine_similarity']=sims
        self.cosine_similarity_df['Frequency_t1'] = self.cosine_similarity_df['Word'].apply(lambda x: word_count_dict_t1[x])
        self.cosine_similarity_df['Frequency_t2'] = self.cosine_similarity_df['Word'].apply(lambda x: word_count_dict_t2[x])

        self.cosine_similarity_df.loc[:,'FrequencyRatio'] = self.cosine_similarity_df['Frequency_t1']/self.cosine_similarity_df['Frequency_t2']
        self.cosine_similarity_df.loc[:,'TotalFrequency'] = self.cosine_similarity_df['Frequency_t1'] + self.cosine_similarity_df['Frequency_t2']

        '''
        self.cosine_similarity_df_sorted = self.cosine_similarity_df.sort_values(by='Cosine_similarity', ascending=True)
        self.cosine_similarity_df_sorted'''

        #Assigning change and no-change labels as initially expected
        self.cosine_similarity_df['semanticDifference']=['default' for i in range(self.cosine_similarity_df.shape[0])]
        self.cosine_similarity_df.loc[self.cosine_similarity_df['Word'].isin(change), 'semanticDifference'] = 'change' 
        self.cosine_similarity_df.loc[self.cosine_similarity_df['Word'].isin(no_change), 'semanticDifference'] = 'no_change'

        self.retrofit_dictkeyvector = dictKeyVector

        # Save into word2vec format for nn comparison
        for t, vocab in [('t1',avg_vec_dict_t1), ('t2',avg_vec_dict_t2)]:
            # vocab = {k:v for k,v in dictKeyVector.items() if t in k}
            self._save_word2vec_format(
                fname = os.path.join(model_output_dir, f'retrofit_vecs_{t}_{self.retrofit_factor}.bin'),
                vocab = vocab,
                vector_size = np.array(vocab[list(vocab.keys())[0]]).shape[0]
            )

        self.model1 = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_output_dir, f'retrofit_vecs_t1_{self.retrofit_factor}.bin'), binary=True)
        self.model2 = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_output_dir, f'retrofit_vecs_t2_{self.retrofit_factor}.bin'), binary=True)

        self.logger.info('Retrofit: Post Process complete')

    def logreg(self, model_output_dir, undersample = True):
        if self.model_type in ['retrofit', 'retro']:
            self.logreg_data = self.cosine_similarity_df.copy()
        else:
            self.logreg_data = self.words_of_interest.copy()

        # drop rows where total frequency is 0
        zero_freq = len(self.logreg_data['TotalFrequency']==0)
        if zero_freq > 0:
            self.logger.info(f'{zero_freq} number of 0 frequency words detected')
            self.logger.info(f"Words are: {self.logreg_data[self.logreg_data['TotalFrequency']==0]['Word'].to_list()}")
            #filter out the rows
            self.logreg_data = self.logreg_data[self.logreg_data['TotalFrequency']>0]
        self.logreg_data['log_freq'] = np.log10(self.logreg_data['TotalFrequency'].astype(float))

        logreg_data_dict = {
            0: ['Cosine_similarity'],
            1: ['Cosine_similarity', 'log_freq'],
            2: ['Cosine_similarity', 'FrequencyRatio'],
            3: ['Cosine_similarity', 'FrequencyRatio', 'log_freq'],
            4: ['mean_cossim'],
            5: ['mean_cossim', 'var_cossim'],
            6: ['mean_cossim', 'log_freq'],
            7: ['mean_cossim', 'var_cossim', 'FrequencyRatio'],
            8: ['mean_cossim', 'FrequencyRatio', 'log_freq'],
            9: ['mean_cossim', 'var_cossim', 'FrequencyRatio', 'log_freq']
        }

        scores_list = []
        for logreg_type in range(max(list(logreg_data_dict.keys()))+1):
            try:
                self.logger.info(f'RUNNING LOGREG. TYPE {logreg_type}')
                # if logreg_type == 0:
                #     X = self.logreg_data['Cosine_similarity'].values.reshape(-1,1)
                # elif logreg_type == 1:
                #     X = self.logreg_data[['Cosine_similarity', 'log_freq']].values.reshape(-1,2)
                # elif logreg_type == 2:
                #     X = self.logreg_data[['Cosine_similarity','FrequencyRatio']].values.reshape(-1,2)
                # elif logreg_type == 3:
                #     X = self.logreg_data[['Cosine_similarity', 'log_freq', 'FrequencyRatio']].values.reshape(-1,3)
                #     # self.logger.info(self.logreg_data)
                if self.model_type =='retrofit':
                    self.logreg_data.to_csv(os.path.join(model_output_dir, f'logreg_df_{self.retrofit_factor}.csv'))
                elif self.model_type == 'speaker_plus':
                    self.logreg_data.to_csv(os.path.join(model_output_dir, f'logreg_df_speaker_plus.csv'))
                else:
                    self.logreg_data.to_csv(os.path.join(model_output_dir, f'logreg_df.csv'))
                X = self.logreg_data[logreg_data_dict[logreg_type]].values.reshape(-1, len(logreg_data_dict[logreg_type]))
                y = self.logreg_data['semanticDifference']

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

                # self.logger.info(X_train)
                kf = logreg.fit(X_train, y_train)

                y_pred = logreg.predict(X_test)

                scoring = {
                    'accuracy' : make_scorer(accuracy_score), 
                    'precision' : make_scorer(precision_score,pos_label='change'),
                    'recall' : make_scorer(recall_score,pos_label='change'), 
                    'f1_score' : make_scorer(f1_score,pos_label='change')
                }

                num_samples = min(np.sum(y=='change'),np.sum(y=='no_change'))
                scores = cross_validate(kf, X, y, cv=min(10, num_samples), scoring=scoring,error_score='raise')

                self.logger.info(f'Accuracy: {scores["test_accuracy"].mean()}')
                self.logger.info(f'Precision, {scores["test_precision"].mean()}')
                self.logger.info(f'Recall, {scores["test_recall"].mean()}')
                self.logger.info(f'F1 Score, {scores["test_f1_score"].mean()}')

                scoresDict = {
                    'Model': f'{self.model_type}',
                    'Basis': 'Cosine Similarity',
                    'Accuracy': f"{scores['test_accuracy'].mean():.3f}",
                    'Precision': f"{scores['test_precision'].mean():.3f}", 
                    'Recall': f"{scores['test_recall'].mean():.3f}",
                    'F1Score': f"{scores['test_f1_score'].mean():.3f}",
                    'Logreg_type': logreg_type,
                    'Input Size': X_train.shape[0],
                    'Train Change Count': np.sum(y_train=='change'),
                    'Train No Change Count': np.sum(y_train=='no_change')
                }
                scores_list.append(scoresDict)
            except Exception as e:
                self.logger.error(e)

        scoresDf = pd.DataFrame.from_records(scores_list)

        self.logger.info(scoresDf)
        if self.model_type == 'retrofit':
            savepath = os.path.join(model_output_dir, f'logreg_{self.retrofit_factor}.csv')
        else:
            savepath = os.path.join(model_output_dir, f'logreg_{self.model_type}.csv')
        scoresDf.to_csv(savepath)

    def nn_comparison(self, model_output_dir, undersample = True):
        self.logger.info(f'Running Nearest Neighbours Comparison')
        neighboursInT1 = []
        neighboursInT2 = []

        if self.model_type in ['retrofit', 'retro']:
            self.words_of_interest = self.cosine_similarity_df.copy()
        elif self.model_type == 'speaker_plus':
            return None

        for row in self.words_of_interest.itertuples():

            if self.model_type in ['speaker', 'retrofit', 'retro']:
                try:
                    x = self.model1.similar_by_word(row.Word,10)
                except KeyError:
                    x = []
                try:
                    y = self.model2.similar_by_word(row.Word,10)
                except KeyError:
                    y = []
            elif self.model_type == 'whole':
                x = self.model1.wv.similar_by_word(row.Word,10) 
                y = self.model2.wv.similar_by_word(row.Word,10)

            x = [tup[0] for tup in x]
            y = [tup[0] for tup in y]
            self.logger.debug(row.Word, x, y)
            neighboursInT1.append(x)
            neighboursInT2.append(y)

        self.words_of_interest['neighboursInT1'] = neighboursInT1
        self.words_of_interest['neighboursInT2'] = neighboursInT2

        self.logger.info(self.words_of_interest)
        self.words_of_interest['overlappingNeighbours'] = self.words_of_interest.apply(lambda row: len(set(row['neighboursInT1']).intersection(set(row['neighboursInT2']))), axis=1)

        self.words_of_interest[self.words_of_interest['semanticDifference']=='change']['overlappingNeighbours'].describe()
        self.words_of_interest[self.words_of_interest['semanticDifference']=='no_change']['overlappingNeighbours'].describe()


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

        num_samples = min(np.sum(y=='change'),np.sum(y=='no_change'))
        scores = cross_validate(kf, X, y, cv=min(10, num_samples), scoring=scoring,error_score='raise')
        accuracy, precision, recall, f1_score_res = [], [], [], []

        self.logger.info(f'Accuracy: {scores["test_accuracy"].mean()}')
        self.logger.info(f'Precision, {scores["test_precision"].mean()}')
        self.logger.info(f'Recall, {scores["test_recall"].mean()}')
        self.logger.info(f'F1 Score, {scores["test_f1_score"].mean()}')

        accuracy.append(scores['test_accuracy'].mean())
        precision.append(scores['test_precision'].mean())
        recall.append(scores['test_recall'].mean())
        f1_score_res.append(scores['test_f1_score'].mean())

        scoresDict = {
            'Model':[f'{self.model_type}'],
            'Basis': ['Cosine Similarity'],
            'Accuracy':accuracy,
            'Precision':precision,
            'Recall':recall,
            'F1Score':f1_score_res,
            'Logreg_type': -1,
            'Input Size': X_train.shape[0],
            'Train Change Count': np.sum(y_train=='change'),
            'Train No Change Count': np.sum(y_train=='no_change')
        }
        scoresDf = pd.DataFrame(scoresDict)
        if self.model_type == 'retrofit':
            savepath = os.path.join(model_output_dir, f'nn_comparison_{self.retrofit_factor}.csv')
        else:
            savepath = os.path.join(model_output_dir, 'nn_comparison.csv')
        scoresDf.to_csv(savepath)

        group1= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'change']
        group2= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'no_change']

        # T Test with 10 neighbours

        summary_neighbours, results_neighbours = rp.ttest(group1= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'change'], group1_name= "change",
                                    group2= self.words_of_interest['overlappingNeighbours'][self.words_of_interest['semanticDifference'] == 'no_change'], group2_name= "no_change")
        # print(summary_neighbours)
        if self.model_type == 'retrofit':
            savepath = os.path.join(model_output_dir, f'nn_comparison_ttest_{self.retrofit_factor}.csv')
        else:
            savepath = os.path.join(model_output_dir, 'nn_comparison_ttest.csv')
        summary_neighbours.to_csv(savepath)



@click.command()
@click.option('--file', '-f', required=True, help='File')
@click.option('--change', '-c', required=True, help='Text file containing words expected to have changed', type=click.File())
@click.option('--no_change', '-nc', required=False, help='Text file containing words NOT expected to have changed', type=click.File())
@click.option('--outdir', required=True, help='Output file directory')
@click.option('--model_output_dir', required=True, help='Outputs after model generation, such as average vectors')
@click.option('--model', required=False, default='whole')
@click.option('--embedding', required=False, default='word')
@click.option('--align/--no-align', default=True)
@click.option('--overlap_req', required=False, default=0.75)
@click.option('--tokenized_outdir', required=False)
@click.option('--min_vocab_size', required=False, type=int)
@click.option('--split_date', required=False, default='2016-06-23 23:59:59')
@click.option('--split_range', required=False, type=int)
@click.option('--retrofit_outdir', required=False)
@click.option('--retrofit_factor', required=False, default='party')
@click.option('--undersample', required=False, is_flag = True)
@click.option('--log_level', required=False, default='INFO')
@click.option('--log_dir', required=False)
@click.option('--log_handler_level', required=False, default='stream')
@click.option('--overwrite_preprocess', required=False, is_flag=True)
@click.option('--overwrite_model', required=False, is_flag=True)
@click.option('--skip_model_check', required=False, is_flag=True)
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
        retrofit_factor,
        model,
        embedding,
        align,
        overlap_req,
        undersample,
        log_level,
        log_dir,
        log_handler_level,
        overwrite_preprocess,
        overwrite_model,
        skip_model_check,
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
    logging.getLogger('smart_open.smart_open_lib').propagate = False

    # Log all the parameters
    logger.info(f'PARAMS - file - {file}')
    logger.info(f'PARAMS - change - {change.name}')
    logger.info(f'PARAMS - no_change - {no_change.name}')
    logger.info(f'PARAMS - outdir - {outdir}')
    logger.info(f'PARAMS - min_vocab_size - {min_vocab_size}')
    logger.info(f'PARAMS - split date -  {split_date}')
    logger.info(f'PARAMS - split range - {split_range}')
    logger.info(f'PARAMS - model_output_dir - {model_output_dir}')
    logger.info(f'PARAMS - tokenized_outdir - {tokenized_outdir}')
    logger.info(f'PARAMS - retrofit_outdir - {retrofit_outdir}')
    logger.info(f'PARAMS - model - {model}')
    logger.info(f'PARAMS - embedding - {embedding}')

    # process change lists
    change_list = []
    for i in change:
        change_list.append(i.strip('\n').strip().lower())
    no_change_list = []
    if no_change:
        for i in no_change:
            no_change_list.append(i.strip('\n').strip().lower())

    # instantiate parliament data handler
    handler = ParliamentDataHandler.from_csv(file, tokenized=False)
    handler.tokenize_data(tokenized_data_dir = tokenized_outdir, overwrite = False)
    date_to_split = split_date
    logger.info(f'SPLITTING BY DATE {date_to_split}')
    handler.split_by_date(date_to_split, split_range)
    logger.info('SPLITTING COMPLETE.')

    # Garbage collection
    gc.collect()

    # unified
    handler.preprocess(
        change = change_list,
        no_change = no_change_list,
        model = model,
        model_output_dir = model_output_dir,
        retrofit_outdir=retrofit_outdir,
        overwrite=overwrite_preprocess
    )

    # Garbage collection
    gc.collect()

    handler.model(
        outdir,
        embedding = embedding, 
        overwrite=overwrite_model,
        skip_model_check = skip_model_check,
        min_vocab_size=min_vocab_size,
        overlap_req=overlap_req,
        align=align
    )

    # Garbage collection
    gc.collect()

    handler.postprocess(
        model_output_dir,
        workers = 10,
        retrofit_factor = retrofit_factor,
        overwrite=overwrite_postprocess
    )
    handler.logreg(model_output_dir, undersample)
    handler.nn_comparison(model_output_dir, undersample)

if __name__ == '__main__':
    main()