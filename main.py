import pandas as pd
from scipy.sparse import vstack
import numpy as np
import models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def rename_yi_large(y):
    y = np.array(y, dtype=object)
    y[y=="accounts/yi-01-ai/models/yi-large"] = "yi-large"
    return y

X_ngram_train = np.load("data/X_ngram_train.npy", allow_pickle=True)
X_ngram_val = np.load("data/X_ngram_val.npy", allow_pickle=True)
X_ngram_test = np.load("data/X_ngram_test.npy", allow_pickle=True)
y_ngram_train = rename_yi_large(np.load("data/y_ngram_train.npy", allow_pickle=True))
y_ngram_val = rename_yi_large(np.load("data/y_ngram_val.npy", allow_pickle=True))
y_ngram_test = rename_yi_large(np.load("data/y_ngram_test.npy", allow_pickle=True))

X_stylometric_train = np.load("data/X_stylometric_train.npy", allow_pickle=True)
X_stylometric_val = np.load("data/X_stylometric_val.npy", allow_pickle=True)
X_stylometric_test = np.load("data/X_stylometric_test.npy", allow_pickle=True)
y_stylometric_train = rename_yi_large(np.load("data/y_stylometric_train.npy", allow_pickle=True))
y_stylometric_val = rename_yi_large(np.load("data/y_stylometric_val.npy", allow_pickle=True))
y_stylometric_test = rename_yi_large(np.load("data/y_stylometric_test.npy", allow_pickle=True))


classifiers = {
    "RandomForest": models.RandomForestClassifierWrapper(n_trees=100),
    "LogisticRegression": models.LogisticRegressionWrapper(max_iter=500),
    "NaiveBayes": models.NaiveBayesClassifierWrapper()
}

# Datasets
datasets = {
    "Ngram_TSV": (X_ngram_train, X_ngram_val, X_ngram_test, y_ngram_train, y_ngram_val, y_ngram_test),
    "Stylometric": (X_stylometric_train, X_stylometric_val, X_stylometric_test, y_stylometric_train, y_stylometric_val, y_stylometric_test)
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

for d_name, (X_tr, X_val, X_te, y_tr, y_val, y_te) in datasets.items():
    print(f"Dataset: {d_name}")

    for clf_name, clf in classifiers.items():
        print(f"Classifier: {clf_name}")

        # Train
        clf.train(X_tr, y_tr)

        # Evaluate on test set
        print("Test set evaluation:")
        clf.evaluate(X_te, y_te)

        # Confusion matrix with correct displayed labels from the test set
        y_pred = clf.predict(X_te)
        labels = np.unique(y_te)
        cm = confusion_matrix(y_te, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45)
        plt.title(f"Confusion Matrix - {clf_name} ({d_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_{clf_name}_{d_name}.png"))
        plt.close()

        x_cv = np.concat([X_tr, X_val])
        y_cv = np.concat([y_tr, y_val])
        # 10-fold CV on full dataset
        print("10-fold cross-validation:")
        clf.cross_validate(x_cv, y_cv, cv=10)

        print("Test set evaluation post cv:")
        clf.evaluate(X_te, y_te)

        # Surrogate tree only for Random Forest
        if clf_name == "RandomForest":
            # feature name(Add proper names in later)
            if d_name == "Stylometric":
                clf.surrogate_tree(
                    x_cv,
                    feature_names=feature_names,
                    class_names=None,
                    max_depth=5,
                    save_name=os.path.join(RESULTS_DIR, f"{d_name}.png")
                )
            else:
                clf.surrogate_tree(
                    x_cv,
                    feature_names=None,
                    class_names=None,
                    max_depth=5,
                    save_name=os.path.join(RESULTS_DIR, f"{d_name}.png")
                )