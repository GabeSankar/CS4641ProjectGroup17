import data_util
import pandas as pd
from scipy.sparse import vstack
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")

train_df, val_df, test_df = data_util.split_by_prompt(df)

# Assume df has at least columns: 'text' and 'label'
train_texts, train_labels = train_df['text'].tolist(), train_df['label'].values
val_texts, val_labels = val_df['text'].tolist(), val_df['label'].values
test_texts, test_labels = test_df['text'].tolist(), test_df['label'].values

# Ngram tsvd dataset
#build vocab over just train rather than train+test
ngram_vocab = data_util.build_ngram_vocab(train_texts, n=2, min_freq=2)
#tsvd
n_components = 100
tsvd = TruncatedSVD(n_components=n_components, random_state=42)

# Build X matrix for n-gram ratio features
#X_ngram = np.array([data_util.ngram_ratio_vector(text, ngram_vocab, n=2) for text in texts])
X_ngram_train = vstack([data_util.ngram_ratio_vector_sparse(text, ngram_vocab, n=2) for text in tqdm(train_texts)])
X_ngram_val = vstack([data_util.ngram_ratio_vector_sparse(text, ngram_vocab, n=2) for text in tqdm(val_texts)])
X_ngram_test = vstack([data_util.ngram_ratio_vector_sparse(text, ngram_vocab, n=2) for text in tqdm(test_texts)])

print("Ngram train/val/test dataset shape:", X_ngram_train.shape, X_ngram_val.shape, X_ngram_test.shape)

X_ngram_train = tsvd.fit_transform(X_ngram_train) # Reuse train principal directions for val and test
X_ngram_val = tsvd.transform(X_ngram_val)
X_ngram_test = tsvd.transform(X_ngram_test)

print("TSVD Ngram train/val/test dataset shape:", X_ngram_train.shape, X_ngram_val.shape, X_ngram_test.shape)

# Stylometric dataset
X_stylometric_train = np.array([data_util.build_stylometric_vector(text, include_pos=True) for text in tqdm(train_texts)])
X_stylometric_val = np.array([data_util.build_stylometric_vector(text, include_pos=True) for text in tqdm(val_texts)])
X_stylometric_test = np.array([data_util.build_stylometric_vector(text, include_pos=True) for text in tqdm(test_texts)])

print("Ngram train/val/test dataset shape:", X_stylometric_train.shape, X_stylometric_val.shape, X_stylometric_test.shape)

np.save("data/X_ngram_train.npy", X_ngram_train)
np.save("data/X_ngram_val.npy", X_ngram_val)
np.save("data/X_ngram_test.npy", X_ngram_test)
np.save("data/y_ngram_train.npy", train_labels)
np.save("data/y_ngram_test.npy", test_labels)
np.save("data/y_ngram_val.npy", val_labels)

np.save("data/X_stylometric_train.npy", X_stylometric_train)
np.save("data/X_stylometric_val.npy", X_stylometric_val)
np.save("data/X_stylometric_test.npy", X_stylometric_test)
np.save("data/y_stylometric_train.npy", train_labels)
np.save("data/y_stylometric_test.npy", test_labels)
np.save("data/y_stylometric_val.npy", val_labels)


