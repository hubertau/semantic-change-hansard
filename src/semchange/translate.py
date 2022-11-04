import os
from google_trans_new import google_translator  
import click
from tqdm import tqdm

@click.command()
@click.option('--infile','-i', help='Input txt file of words', required=True, type=click.File('r'))
@click.option('--outlang', '-o', help='Output Language', required=True)
def main(infile, outlang):
    words = infile.readlines()
    words = [i.strip('\n').strip() for i in words]
    words = [i for i in words if len(i)>0]
    translator = google_translator()
    translations = []
    for word in tqdm(words):
        translations.append(translator.translate(word, lang_tgt=outlang))
    outfile = f"{os.path.split(infile.name)[-1].split('.')[0]}_{outlang}.txt"
    outfile = os.path.join(os.path.split(infile.name)[0], outfile)
    with open(outfile, 'w') as f:
        for t in translations:
            if isinstance(t, list):
                for i in t:
                    f.write(i)
                    f.write('\n')
            elif isinstance(t, str):
                f.write(t)
                f.write('\n')
    print(f'Saved to {outfile}')

if __name__=='__main__':
    main(
        [
            "-i",
            "/Users/hubert/Drive/DPhil/Projects/2022-08a-Semantic_Change/semantic-change-hansard/data/02_intermediate/Corp_HouseOfCommons_nochange.txt",
            "-o",
            "sv"
        ]
    )