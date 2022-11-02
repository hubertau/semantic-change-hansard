import pickle

with open('./df_t1_Plaid Cymru.pkl','rb') as f:
    df = pickle.load(f)
    print(df.sentence)
    print(df.lemmatized)
