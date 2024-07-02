import pandas as pd
import os


def main():

    outfile = None

    data_file = os.getenv('DATAFILE')
    if not data_file:
        raise FileNotFoundError
    
    # OUT

    data = pd.read_parquet(data_file)

    TOP_PARTIES_BY_SPEECH_COUNT = data.groupby(['party']).count().sort_values(by='text', ascending=False).head(5).index


    TOP_SPEAKERS_IN_TOP_PARTIES = data.query('party in @TOP_PARTIES_BY_SPEECH_COUNT').groupby(['party', 'speaker']).count().sort_values(by=['party','text'], ascending=False).groupby('party').head()['text'].reset_index()['speaker']

    # with open()
# 


if __name__ == '__main__':

    main()