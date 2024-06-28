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
from datetime import datetime
import sys

def main():

    logger.level(os.getenv('LOGLEVEL', 'INFO'))
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")

    # Set a random seed
    random_seed = 42
    random.seed(random_seed)

    logger.info(f'CUDA: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Layers to consider for embedding
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
        logger.warning(f'File found at {hdf5_file}, not overwriting. Ending...')
        return None

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    logger.info(f"Attempting to load in model: {os.getenv('MODEL')}")
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('MODEL'), cache_dir=f"{os.getenv('CACHEDIR')}")
    model = AutoModel.from_pretrained(
        os.getenv('MODEL'),
        cache_dir=f"{os.getenv('CACHEDIR')}", output_hidden_states=True,
    )
    model.to(device)
    logger.info(f"Loaded in model: {os.getenv('MODEL')}")

    # Example data
    data = pd.read_parquet(os.getenv('DATAFILE'))
    logger.info(f"Loaded in data from: {os.getenv('DATAFILE')}")

    # specify the specific ids in here
    text_query = data.query(f'speaker == "{os.getenv("SPEAKER")}"').sort_index()

    # Retrieve all texts and their associated dates
    texts = text_query['text'].apply(str.lower).to_list()
    dates = text_query['date'].to_list()

    # Encode documents
    encoding = tokenizer.batch_encode_plus(
        texts,                      # List of input documents
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

    all_document_embeddings = []

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

            # Aggregate embeddings for each document (mean pooling)
            document_embeddings = torch.mean(output, dim=1)
            all_document_embeddings.append(document_embeddings.cpu())

    # Concatenate all document embeddings
    all_document_embeddings = torch.cat(all_document_embeddings, dim=0)
    texts_array = np.array(texts, dtype=h5py.string_dtype(encoding='utf-8'))
    dates_array = np.array([date.isoformat() for date in dates], dtype=h5py.string_dtype(encoding='utf-8'))

    # Store embeddings, texts, and dates
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('embeddings', data=all_document_embeddings.numpy())
        f.create_dataset('texts', data=texts_array)
        f.create_dataset('dates', data=dates_array)

    logger.info(f"Embeddings stored in {hdf5_file}")

if __name__ == '__main__':
    main()