import os
import data_util
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

# preparation
os.makedirs("final_data", exist_ok=True)
df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
train_df, val_df, test_df = data_util.split_by_prompt(df)
train_texts = train_df['text'].tolist()
val_texts = val_df['text'].tolist()
test_texts = test_df['text'].tolist()

print(f"train: {len(train_texts)}, val: {len(val_texts)}, test: {len(test_texts)}")

# configs to sweep over for lex
ngram_ranges = [(1,1), (1,2), (2,2)]
tfidf_flags = [False, True]
ranks = [50, 100, 300]

configs = list(product(ngram_ranges, tfidf_flags, ranks))
# loop thru each config, for train and val
# only train and val bc test won't be used since we fit from train+val later for test
for i, (ngram_range, use_tfidf, tsvd_rank) in enumerate(configs, 1):
    ngram_str = f"{ngram_range[0]}-{ngram_range[1]}"
    # some logging to track progress per since this will take a while
    print(f"[{i}/{len(configs)}] lex ngram={ngram_str} tfidf={use_tfidf} rank={tsvd_rank}")
    vec, svd, X_train = data_util.build_lex_pipeline(train_texts, ngram_range, use_tfidf, tsvd_rank)
    X_val = svd.transform(vec.transform(val_texts))
    base = f"final_data/X_lex_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}"
    np.save(f"{base}_train.npy", X_train)
    np.save(f"{base}_val.npy", X_val)

print("finished lex features now building stylometric features")

# one shot
print("starting train")
X_sty_train = np.array([data_util.build_stylometric_vector(t, include_pos=True) for t in tqdm(train_texts, desc="train")])
print("finished train")
print("starting val")
X_sty_val = np.array([data_util.build_stylometric_vector(t, include_pos=True) for t in tqdm(val_texts, desc="val")])
print("finished val")
print("starting test")
X_sty_test = np.array([data_util.build_stylometric_vector(t, include_pos=True) for t in tqdm(test_texts, desc="test")])
print("finished test")

np.save("final_data/X_stylometric_train.npy", X_sty_train)
np.save("final_data/X_stylometric_val.npy", X_sty_val)
np.save("final_data/X_stylometric_test.npy", X_sty_test)

print(f"saved all files to final_data/")
