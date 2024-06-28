import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import h5py
import torch
import json
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from dataclasses import dataclass
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from umap import UMAP
import sys
import glob

BASE_PATH = Path(os.getenv('BASE_PATH'))

DATA_PATH = Path(os.getenv('DATA_PATH'))
EMBED_PATH = Path(os.getenv('EMBED_PATH'))
GRAPH_PATH = Path(os.getenv('GRAPH_PATH'))
INTERMEDIATE_PATH = Path(os.getenv('INTERMEDIATE_PATH'))

WORDS_FILE = Path(os.getenv('WORDS_FILE'))

# reference word for 
# REF_WORD = Path(os.getenv('REF_WORD'))

sns.set_theme('paper')

@dataclass
class speaker_embedding:
    embeddings: dict
    times: list
    time_indices: list
    words: list
    freq: str
    speaker: str

def get_speaker(file_path):
    return ' '.join(file_path.name.split('_')[:2])

def get_words_used(times, words):
    words_used = Counter()

    for idx, _ in enumerate(times):
        for _, word_bytes in enumerate(words[idx]):
            words_used[word_bytes]+=1

    return words_used

def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        embeddings = {}
        times = []
        time_indices = []
        words = []
        for key in f['embeddings'].keys():
            embeddings[key] = f['embeddings'][key][:]
            times.append(f['times'][key][:])
            time_indices.append(key)
            words.append([word_bytes.decode('utf-8') for word_bytes in f['words'][key][:]])
        frequency = f.attrs['frequency']
    speaker = get_speaker(file_path)

    out = speaker_embedding(
        embeddings=embeddings,
        times = times,
        time_indices = time_indices,
        words = words,
        freq = frequency,
        speaker = speaker
    )

    return out

def _join_woi_with_suffixes(woi, suffix_list):
    return woi + ''.join(suffix_list) if suffix_list else woi

# Set up functions for alignment
def retrieve_intersection(prev_embed, curr_embed, words=None):
    # Get the vocab for each model
    vocab_m1 = set(prev_embed[0])
    vocab_m2 = set(curr_embed[0])

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (prev_embed,curr_embed)

    prev_indices = []
    prev_words = []
    curr_indices = []
    curr_words = []
    for i, x in enumerate(prev_embed[0]):
        if x in common_vocab:
            prev_indices.append(i)
            prev_words.append(x)
            curr_words.append(x)
            curr_index = curr_embed[0].index(x)
            curr_indices.append(curr_index)

    return (
        (prev_words, prev_embed[1][np.array(prev_indices)]),
        (curr_words, curr_embed[1][np.array(curr_indices)])
    )

def _smart_procrustes_align_gensim(base_embed, other_embed, words=None):
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
    # in_base_embed, in_other_embed = self._intersection_align_gensim(base_embed, other_embed, words=words)

    # in_base_embed.wv.fill_norms(force=True)
    # in_other_embed.wv.fill_norms(force=True)

    # print(5)

    # get the (normalized) embedding matrices
    # base_vecs = in_base_embed.wv.get_normed_vectors()
    # other_vecs = in_other_embed.wv.get_normed_vectors()
    base_vecs = np.divide(base_embed, np.linalg.norm(base_embed).reshape(-1,1))
    other_vecs = np.divide(other_embed, np.linalg.norm(other_embed).reshape(-1,1))


    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    # other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)

    return ortho

def main():

    logger.level(os.getenv('LOGLEVEL', 'INFO'))
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")

    logger.info(f'CUDA: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # files = glob.glob(str(EMBED_PATH / '*.h5'))
    # files = [Path(i) for i in files]

    pair = os.getenv('SPEAKER_PAIR').split(',')

    files = [
        EMBED_PATH / f'{pair[0].replace(' ', '_')}.h5',
        EMBED_PATH / f'{pair[1].replace(' ', '_')}.h5'
    ]

    for i in files:
        logger.info(f'Embedding file to be used: {i.stem}.h5')

    # extract embeddings from files
    extracted_embeds = {get_speaker(file): load_embeddings(file) for file in files}

    # get wois
    with open(WORDS_FILE, 'r') as f:
        wois = json.load(f)

    # Retrieve model with most overlap:
    overlaps = Counter()

    logger.info('Extracting most overlapping embed...')
    # process overlap
    for speaker, e in extracted_embeds.items():
        # print(speaker)
        for idx, interval in enumerate(e.times):

            # normalise embeddings
            curr_embed = e.embeddings[e.time_indices[idx]]
            curr_embed = np.divide(curr_embed,np.linalg.norm(curr_embed, axis=1).reshape(-1,1))

            # add another for loop
            for idx2, _ in enumerate(e.times[idx+1:]):

                idx2+=idx

                # perform alignment:
                next_embed = e.embeddings[e.time_indices[idx2]]
                next_embed = np.divide(next_embed,np.linalg.norm(next_embed, axis=1).reshape(-1,1))

                result = retrieve_intersection(
                    (e.words[idx], curr_embed),
                    (e.words[idx2], next_embed),
                )

                overlaps[(speaker, idx)] += len(result[0][0])
                overlaps[(speaker, idx2)] += len(result[0][0])
    logger.info('Done.')

    # get reference embed from most overlaps
    ref_embed = overlaps.most_common(1)[0][0]
    # print(f'{ref_embed}')
    ref_embed_obj = extracted_embeds[ref_embed[0]]
    ref_embed_idx = ref_embed[1]


    # with open(os.getenv('REF_WORDLIST'), 'r') as f:
    #     ref_woi_list = f.readlines()
    #     ref_woi_list = [i.replace('\n', '') for i in ref_woi_list]
    #     ref_woi_dict = {
    #         i: wois[i] for i in ref_woi_list
    #     }

    # ref_woi_embed_dict = {}

    # model = AutoModel.from_pretrained(
    #     'FacebookAI/xlm-roberta-large-finetuned-conll03-english',
    #     output_hidden_states=True
    # )
    # tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large-finetuned-conll03-english')

    # for w, ws in ref_woi_dict:
    #     # Encode the word with suffixes
    #     joined_word = _join_woi_with_suffixes(w, ws)
    #     encoding = tokenizer.encode(joined_word, return_tensors='pt')

    #     # Get the model's output
    #     outputs = model(encoding)
    #     states = outputs.hidden_states

    #     # Sum the embeddings from the last 4 layers
    #     layers = [-4, -3, -2, -1]
    #     output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    #     # Decode the tokens to find the positions of sub-word tokens
    #     decoded_tokens = tokenizer.convert_ids_to_tokens(encoding[0])
    #     subword_indices = [i for i, token in enumerate(decoded_tokens) if token.startswith('▁') or (token != decoded_tokens[0] and token != decoded_tokens[-1])]

    #     # Sum the embeddings of the sub-word tokens
    #     ref_vec = output[subword_indices].sum(dim=0).detach().numpy()

    #     # Store the resulting embedding in the dictionary
    #     ref_woi_embed_dict[joined_word] = ref_vec

    # logger.info('Retrieving reference vec...')

    # if ref_woi in ref_embed_obj.words[ref_embed_idx]:

    #     ref_vec = ref_embed_obj.embeddings[ref_embed_obj.time_indices[ref_embed_idx]][ref_embed_obj.words[ref_embed_idx].index(ref_woi)]

    #     for suf in ref_woi_suffixes:
    #         ref_vec += ref_embed_obj.embeddings[ref_embed_obj.time_indices[ref_embed_idx]][ref_embed_obj.words[ref_embed_idx].index(suf)] 

    #     ref_vec = ref_vec.reshape(1,-1)

    # else:
    #     raise ValueError(f'Reference word {ref_woi} not in most overlapping embedding')

    # logger.info('Done')


    # define start objects
    start_times = []
    end_times = []
    # cos = []
    # cos = {}
    woi_embeds = []
    speakers = []
    corresponding_words = []


    logger.info(f'Extracting word embeddings...')
    # perform extraction with alignment
    counter = 1
    for woi, woi_suffixes in wois.items():
        COMBINED_WORD = _join_woi_with_suffixes(woi, woi_suffixes)
        if COMBINED_WORD in corresponding_words:
            continue

        # FOR DEBUGGING
        # if counter > 2:
        #     break

        logger.info(f'Processing {counter} of {len(wois)}: {COMBINED_WORD}')
        counter += 1

        for speaker, e in extracted_embeds.items():

            for idx, interval in enumerate(e.times):

                if len(set(e.words[idx])) != len(e.words[idx]):
                    logger.error('SOMETHING IS WRONG')
                    break

                if woi not in set(e.words[idx]):
                    continue

                # retrieve the current embedding from the reference object and normalise it
                curr_embed = e.embeddings[e.time_indices[idx]]
                curr_embed = np.divide(curr_embed,np.linalg.norm(curr_embed, axis=1).reshape(-1,1))

                # align with reference embedding
                result = retrieve_intersection(
                    (ref_embed_obj.words[ref_embed_idx], ref_embed_obj.embeddings[ref_embed_obj.time_indices[ref_embed_idx]]),
                    (e.words[idx], curr_embed)
                )
                ortho = _smart_procrustes_align_gensim(result[0][1], result[1][1])
                curr_embed = (curr_embed).dot(ortho)

                # retrieve current word index
                current_woi_index = e.words[idx].index(woi)

                # extract the word embedding
                woi_embed = curr_embed[current_woi_index]

                # and the same for any suffixes
                for suffix in woi_suffixes:
                    if suffix in e.words[idx]:
                        current_woi_suffix_index = e.words[idx].index(suffix)
                        woi_embed += curr_embed[current_woi_suffix_index] 

                # append the embedding
                woi_embeds.append(woi_embed)

                # track times
                start_times.append(pd.to_datetime(interval[0].decode('utf-8')))
                end_times.append(pd.to_datetime(interval[1].decode('utf-8')))

                # who spoke it?
                speakers.append(speaker)

                # cosine similarity to our reference word
                # for ref_w, ref_w_embed in ref_woi_dict.items():
                #     cos[ref_w].append(cosine_similarity(ref_w_embed.reshape(1, -1), woi_embed.reshape(1, -1)).item())

                # and the word itself
                corresponding_words.append(COMBINED_WORD)

    logger.info(f'Running UMAP')
    X_embedded = UMAP(
        n_components=2,
        n_neighbors=3,
        min_dist=0
    ).fit_transform(np.stack(woi_embeds))

    # COMBINED_REF_WORD = _join_woi_with_suffixes(ref_woi, ref_woi_suffixes)

    SPEAKER_LIST_FOR_FILENAME = list(set(speakers))
    SPEAKER_LIST_FOR_FILENAME = [i.split(' ')[-1] for i in SPEAKER_LIST_FOR_FILENAME]
    SPEAKER_LIST_FOR_FILENAME = '_'.join(SPEAKER_LIST_FOR_FILENAME)
    PLOT_DF_FILENAME = DATA_PATH / '../03_processed' / f'umap_GROUP_covid_data_{SPEAKER_LIST_FOR_FILENAME}.parquet'
    logger.info(f'Saving to {PLOT_DF_FILENAME.stem}')

    PLOT_DF = pd.DataFrame({
        'speaker': speakers,
        'words': corresponding_words,
        'UMAP Component 1': X_embedded[:,0],
        'UMAP Component 2': X_embedded[:,1],
        'month': start_times,
        'embeds': woi_embeds
    })

    # for ref_w, vals in cos.items():
        # PLOT_DF[f'cos_{ref_w}'] = vals

    PLOT_DF.to_parquet(PLOT_DF_FILENAME)


if __name__=='__main__':
    main()