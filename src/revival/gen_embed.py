import numpy as np
import pandas as pd
from pathlib import Path
import os
from loguru import logger
import random
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sys

def generate_time_intervals(start_date, end_date, frequency='3M'):
    intervals = []
    current_date = start_date
    while current_date <= end_date:
        intervals.append(current_date)
        if frequency.endswith('M'):
            months = int(frequency[:-1])
            current_date += timedelta(days=30*months)
        elif frequency.endswith('D'):
            days = int(frequency[:-1])
            current_date += timedelta(days=days)
        # Add more conditions for other frequencies if needed
    return intervals

def main():

    logger.level(os.getenv('LOGLEVEL', 'INFO'))
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")

    # Set a random seed
    random_seed = 42
    random.seed(random_seed)

    logger.info(f'CUDA: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Layers to consider
    layers = [-4, -3, -2, -1]

    if not os.getenv('SPEAKER'):
        raise ValueError('Please provide a speaker.')
    logger.info(f'SPEAKER is: {os.getenv("SPEAKER")}')

    # Prepare storage
    hdf5_file = Path(os.getenv('SAVEFOLDER'))/f"{os.getenv('SPEAKER')}_embeddings.h5".replace(" ", "_")
    assert os.path.isdir(Path(os.getenv('SAVEFOLDER')))
    logger.info(f'Embeddings to be saved to: {hdf5_file}')

    OVERWRITE = bool(os.getenv('OVERWRITE'))
    if os.path.isfile(hdf5_file) and not OVERWRITE:
        logger.warning(f'File found at {hdf5_file}, not overwriting. Endings...')
        return None

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    logger.info(f"Attempting to load in model: {os.getenv('MODEL')}")
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL'), cache_dir = f"{os.getenv('CACHEDIR')}")
    model = AutoModel.from_pretrained(
        os.getenv('MODEL'),
        cache_dir = f"{os.getenv('CACHEDIR')}", output_hidden_states=True,
    )
    model.to(device)
    logger.info(f"Loaded in model: {os.getenv('MODEL')}")

    # Example data
    data = pd.read_parquet(os.getenv('DATAFILE'))
    logger.info(f"Loaded in data from: {os.getenv('DATAFILE')}")

    # specify the specific ids in here
    text_query = data.query(f'speaker == "{os.getenv("SPEAKER")}"').sort_index()
    # text_data = text_query['text'].to_list()

    # text_data = ["Your text data here..."]
    # max_length = tokenizer.model_max_length

    # Parameters
    frequency = os.getenv('INTERVAL','1M')  # Change this as needed
    logger.info(f'Interval set at: {frequency}')
    start_date = datetime(1988, 1, 1)
    end_date = datetime(2020, 1, 1)
    time_intervals = generate_time_intervals(start_date, end_date, frequency=frequency)


    with h5py.File(hdf5_file, 'w') as f:
        f.attrs['frequency'] = frequency
        embedding_group = f.create_group('embeddings')
        words_group = f.create_group('words')
        times_group = f.create_group('times')

        for time_idx, time_point in enumerate(time_intervals[:-1]):
            next_time_point = time_intervals[time_idx + 1]
            logger.info(f'Processing: {time_point} to {next_time_point}')

            # Retrieve texts within the time interval
            mask = (text_query['date'] >= time_point) & (text_query['date'] < next_time_point)
            texts_for_time = text_query.loc[mask, 'text'].apply(str.lower).to_list()

            if not texts_for_time:
                continue

            # Split and encode text
            # chunked_texts = []
            # for text in texts_for_time:
            #     chunked_texts.extend(chunk_text(text, max_length))

            encoding = tokenizer.batch_encode_plus(
                texts_for_time,                 # List of input text chunks
                padding=True,                  # Pad to the maximum sequence length
                truncation=True,               # Truncate to the maximum sequence length if necessary
                return_tensors='pt',           # Return PyTorch tensors
                add_special_tokens=True        # Add special tokens CLS and SEP
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # input_ids.to(device)
            # attention_mask.to(device)

            dataset = TensorDataset(input_ids, attention_mask)
            batch_size = 32
            dataloader = DataLoader(dataset, batch_size=batch_size)

            # all_embeddings = []

            all_words = defaultdict(list)

            logger.info('Doing batch encoding')
            with torch.no_grad():
                for batch in dataloader:
                    input_ids_batch, attention_mask_batch = batch
                    logger.debug(input_ids_batch.shape)

                    input_ids_batch = input_ids_batch.to(device)
                    attention_mask_batch = attention_mask_batch.to(device)

                    outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
                    states = outputs.hidden_states
                    # Stack and sum all requested layers
                    output = torch.stack([states[i] for i in layers]).sum(0)
                    logger.debug(output.shape)
                    # Only select the tokens that constitute the requested word
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            try:
                                all_words[input_ids_batch[i][j].item()].append(output[i][j])
                            except:
                                logger.warning(input_ids_batch.shape)
                                logger.warning(output.shape)
                                logger.warning((i,j))
                                raise ValueError

            words_final = {tokenizer.decode(k): torch.mean(torch.stack(v),axis=0) for k, v in all_words.items()}

            all_embeddings = torch.zeros(len(words_final), words_final[list(words_final.keys())[0]].shape[0])
            words_list = []

            for i, (k, v) in enumerate(words_final.items()):
                all_embeddings[i] = v
                words_list.append(k)

            words_array = np.array(words_list, dtype=h5py.string_dtype(encoding='utf-8'))

            # Store embeddings, words, and times
            embedding_group.create_dataset(f"time_{time_idx}", data=all_embeddings.numpy())
            words_group.create_dataset(f"time_{time_idx}", data=words_array)
            times_group.create_dataset(f"time_{time_idx}", data=np.string_([time_point.isoformat(), next_time_point.isoformat()]))

    logger.info(f"Embeddings stored in {hdf5_file}")

if __name__ == '__main__':
    main()
