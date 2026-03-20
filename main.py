from sklearn.model_selection import train_test_split
import data_util
import pandas as pd
from scipy.sparse import vstack
import numpy as np
from sklearn.decomposition import TruncatedSVD
import models
from tqdm import tqdm

# df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
# df = data_util.flip_dataframe(df)

# Assume df has at least columns: 'text' and 'label'
# texts = df['text'].tolist()
# labels = df['label'].values  # target

# # Ngram tsvd dataset
# #vocab
# ngram_vocab = data_util.build_ngram_vocab(texts, n=2, min_freq=2)
# #tsvd
# n_components = 100
# tsvd = TruncatedSVD(n_components=n_components, random_state=42)

# # Build X matrix for n-gram ratio features
# #X_ngram = np.array([data_util.ngram_ratio_vector(text, ngram_vocab, n=2) for text in texts])
# X_ngram = vstack([data_util.ngram_ratio_vector_sparse(text, ngram_vocab, n=2) for text in texts])
# y_ngram = labels.copy()

# X_ngram_reduced = tsvd.fit_transform(X_ngram)

# print("Ngram dataset shape:", X_ngram.shape)

# print("TSVD Ngram dataset shape:", X_ngram_reduced)

# # Stylometric dataset
# stylometric_vectors = []
# for text in tqdm(texts):
#     stylometric_vectors.append(data_util.build_stylometric_vector(text, include_pos=True))

# X_stylometric = np.array(stylometric_vectors)
# y_stylometric = labels.copy()

# print("Stylometric dataset shape:", X_stylometric.shape)

# np.save("X_ngram.npy", X_ngram_reduced)
# np.save("y_ngram.npy", y_ngram)

# np.save("X_stylometric.npy", X_stylometric)
# np.save("y_stylometric.npy", y_stylometric)

X_ngram_reduced, y_ngram = np.load("X_ngram.npy",  allow_pickle=True), np.load("y_ngram.npy",  allow_pickle=True)
X_stylometric, y_stylometric = np.load("X_stylometric.npy",  allow_pickle=True), np.load("y_stylometric.npy",  allow_pickle=True)

#train test and 10cv
X_ngram_train, X_ngram_test, y_ngram_train, y_ngram_test = train_test_split(
    X_ngram_reduced, y_ngram, test_size=0.2, random_state=42
)

X_styl_train, X_styl_test, y_styl_train, y_styl_test = train_test_split(
    X_stylometric, y_stylometric, test_size=0.2, random_state=42
)



classifiers = {
    "RandomForest": models.RandomForestClassifierWrapper(n_trees=100),
    "LogisticRegression": models.LogisticRegressionWrapper(max_iter=500),
    "NaiveBayes": models.NaiveBayesClassifierWrapper()
}

# Datasets
datasets = {
    "Ngram_TSV": (X_ngram_train, X_ngram_test, y_ngram_train, y_ngram_test, X_ngram_reduced, y_ngram),
    "Stylometric": (X_styl_train, X_styl_test, y_styl_train, y_styl_test, X_stylometric, y_ngram)
}

feature_names = [
    "avg_sent_len",
    "var_sent_len",
    "avg_word_len",
    "ttr",
    "punct_ratio",
    "punct_.",
    "punct_,",
    "punct_!",
    "punct_?",
    "punct_;",
    "punct_:",
    "pos_NN",
    "pos_VB",
    "pos_JJ",
    "pos_RB",
    "pos_DT",
    "pos_IN",
    "pos_PRP"
]

for d_name, (X_tr, X_te, y_tr, y_te, X_full, y_full) in datasets.items():
    print(f"Dataset: {d_name}")
    
    for clf_name, clf in classifiers.items():
        print(f"Classifier: {clf_name}")
        
        # Train
        clf.train(X_tr, y_tr)
        
        # Evaluate on test set
        print("Test set evaluation:")
        clf.evaluate(X_te, y_te)
        
        # 10-fold CV on full dataset
        print("10-fold cross-validation:")
        clf.cross_validate(X_full, y_full, cv=10)
        
        print("Test set evaluation post cv:")
        clf.evaluate(X_te, y_te)
        
        # Surrogate tree only for Random Forest
        if clf_name == "RandomForest":
            # feature name(Add proper names in later)
            if d_name == "Stylometric":
                clf.surrogate_tree(
                    X_full,
                    feature_names=feature_names,
                    class_names=None,
                    max_depth=20,
                    save_name=f"{d_name}.png"
                )
            else:
                clf.surrogate_tree(
                    X_full,
                    feature_names=None,
                    class_names=None,
                    max_depth=20,
                    save_name=f"{d_name}.png"
                )
              
