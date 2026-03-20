from sklearn.model_selection import train_test_split
import data_util
import pandas as pd
from scipy.sparse import vstack
import numpy as np
from sklearn.decomposition import TruncatedSVD
import models

df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
df = data_util.flip_dataframe(df)

# Assume df has at least columns: 'text' and 'label'
texts = df['text'].tolist()
labels = df['label'].values  # target

# Ngram tsvd dataset
#vocab
ngram_vocab = data_util.build_ngram_vocab(texts, n=2, min_freq=2)
#tsvd
n_components = 100
tsvd = TruncatedSVD(n_components=n_components, random_state=42)

# Build X matrix for n-gram ratio features
#X_ngram = np.array([data_util.ngram_ratio_vector(text, ngram_vocab, n=2) for text in texts])
X_ngram = vstack([data_util.ngram_ratio_vector_sparse(text, ngram_vocab, n=2) for text in texts])
y_ngram = labels.copy()

X_ngram_reduced = tsvd.fit_transform(X_ngram)

print("Ngram dataset shape:", X_ngram.shape)

print("TSVD Ngram dataset shape:", X_ngram_reduced)

# Stylometric dataset
X_stylometric = np.array([data_util.build_stylometric_vector(text, include_pos=True) for text in texts])
y_stylometric = labels.copy()

print("Stylometric dataset shape:", X_stylometric.shape)

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
    "Ngram_TSV": (X_ngram_train, X_ngram_test, y_ngram_train, y_ngram_test, X_ngram_reduced, labels),
    "Stylometric": (X_styl_train, X_styl_test, y_styl_train, y_styl_test, X_stylometric, labels)
}

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
                feature_names = [f"feature_{i}" for i in range(X_full.shape[1])]
            else:
                feature_names = None
                
            clf.surrogate_tree(
                X_full,
                feature_names=feature_names,
                class_names=None,
                max_depth=20
            )
