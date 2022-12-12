import click
from semchange import ParliamentDataHandler
import logging
import os
import pickle
import gensim
from collections import Counter
import pandas as pd

@click.command()
@click.option('--file', '-f', required=True, help='File')
@click.option('--change', '-c', required=True, help='Text file containing words expected to have changed', type=click.File())
@click.option('--no_change', '-nc', required=False, help='Text file containing words NOT expected to have changed', type=click.File())
@click.option('--outdir', required=True, help='Output file directory')
@click.option('--model_output_dir', required=True, help='Outputs after model generation, such as average vectors')
@click.option('--model', required=False, default='whole')
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

    # unified
    handler.preprocess(
        change = change_list,
        no_change = no_change_list,
        model = model,
        model_output_dir = model_output_dir,
        retrofit_outdir=retrofit_outdir,
        overwrite=overwrite_preprocess
    )
    handler.model(
        outdir,
        overwrite=overwrite_model,
        skip_model_check = skip_model_check,
        min_vocab_size=min_vocab_size,
        overlap_req=overlap_req,
        align=align
    )

    logger.info(f'MODELLING COMPLETE. NOW EXTRACTING WORDS...')
    change_dict = {k: Counter() for k in change}
    for model_path in handler.speaker_saved_models:
        model = gensim.models.Word2Vec.load(model_path)
        for word in change:
            if word in model.wv.index_to_key:
                change_dict[word]['self'] += 1
                words_to_add = model.similar_by_word(word, 10)
                for word_to_add in words_to_add:
                    if word_to_add not in change:
                        change_dict[word][word_to_add] += 1
    with open(os.path.join(model_output_dir, f'words.pkl'), 'wb') as f:
        pickle.dump(change_dict, f)
    colnames = [list(i.keys()) for i in change_dict.values()]
    colnames = [item for sublist in colnames for item in sublist]
    colnames = list(set(colnames))
    out = pd.DataFrame.from_dict(orient='index')
    out.to_csv(os.path.join(model_output_dir, f'words.csv'))

if __name__ == '__main__':
    main()