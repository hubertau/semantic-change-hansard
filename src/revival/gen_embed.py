import numpy as np
import pandas as pd
from pathlib import Path
import os
from loguru import logger
from tqdm.notebook import tqdm
import random
import torch
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import click
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import h5py
import numpy as np
from datetime import datetime, timedelta

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

    # Set a random seed
    random_seed = 42
    random.seed(random_seed)

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL'))
    model = AutoModel.from_pretrained(os.getenv('MODEL'))
    logger.info(f"Loaded in model: {os.getenv('MODEL')}")

    # Example data
    data = pd.read_parquet(os.getenv('DATA'))
    logger.info(f"Loaded in data from: {os.getenv('DATA')}")

    # specify the specific ids in here
    text_query = data.query(f'speaker == {os.getenv("SPEAKER")}').sort_index()
    # text_data = text_query['text'].to_list()

    # text_data = ["Your text data here..."]
    # max_length = tokenizer.model_max_length

    # Parameters
    frequency = os.getenv('INTERVAL','M')  # Change this as needed
    logger.info(f'Interval set at: {frequency}')
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 1, 1)
    time_intervals = generate_time_intervals(start_date, end_date, frequency=frequency)

    # Prepare storage
    hdf5_file = Path(os.get_env('SAVEFOLDER'))/f"{os.getenv('SPEAKER')}_embeddings.h5"
    assert os.isdir(Path(os.get_env('SAVEFOLDER')))
    logger.info(f'Embeddings to be saved to: {hdf5_file}')

    with h5py.File(hdf5_file, 'w') as f:
        f.attrs['frequency'] = frequency
        embedding_group = f.create_group('embeddings')
        words_group = f.create_group('words')
        times_group = f.create_group('times')

        for time_idx, time_point in enumerate(time_intervals[:-1]):
            next_time_point = time_intervals[time_idx + 1]

            # Retrieve texts within the time interval
            mask = (text_query['date'] >= time_point) & (text_query['date'] < next_time_point)
            texts_for_time = text_query.loc[mask, 'text'].tolist()

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

            dataset = TensorDataset(input_ids, attention_mask)
            batch_size = 32
            dataloader = DataLoader(dataset, batch_size=batch_size)

            all_embeddings = []

            with torch.no_grad():
                for batch in dataloader:
                    input_ids_batch, attention_mask_batch = batch
                    outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
                    word_embeddings = outputs.last_hidden_state
                    all_embeddings.append(word_embeddings)

            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

            # Store embeddings, words, and times
            embedding_group.create_dataset(f"time_{time_idx}", data=all_embeddings)
            words_group.create_dataset(f"time_{time_idx}", data=np.array(texts_for_time, dtype='S'))
            times_group.create_dataset(f"time_{time_idx}", data=np.string_([time_point.isoformat(), next_time_point.isoformat()]))

    logger.info(f"Embeddings stored in {hdf5_file}")

if __name__ == '__main__':
    main()
