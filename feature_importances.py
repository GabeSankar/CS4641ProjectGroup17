import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_util
import models

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


# prep
DATA_DIR = "final_data"
IMPORTANCE_DIR = "final_results/testing_winners/feature_importances"
SURROGATE_DIR = "final_results/testing_winners/surrogate_trees"
os.makedirs(IMPORTANCE_DIR, exist_ok=True)
os.makedirs(SURROGATE_DIR, exist_ok=True)

STYLOMETRIC_FEATURE_NAMES = [
    "avg_sent_len", "var_sent_len", "avg_word_len", "ttr", "punct_ratio",
    "punct_.", "punct_,", "punct_!", "punct_?", "punct_;", "punct_:",
    "pos_NN", "pos_VB", "pos_JJ", "pos_RB", "pos_DT", "pos_IN", "pos_PRP",
]

# lol
def rename_yi_large(y):
    y = np.array(y, dtype=object)
    y[y == "accounts/yi-01-ai/models/yi-large"] = "yi-large"
    return y


def make_rf(hyperparameter_value):
    # max_features is either 'sqrt' string or a float
    try:
        v = float(hyperparameter_value)
    except ValueError:
        v = hyperparameter_value
    return models.RandomForestClassifierWrapper(max_features=v)


def save_importance(run_id, feature_names, importances, top_n, label):
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(IMPORTANCE_DIR, f"{run_id}.csv")
    importance_df.to_csv(csv_path, index=False)

    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    ax.barh(top['feature'], top['importance'])
    ax.set_xlabel(label)
    ax.set_title(f"{label} - {run_id}", fontsize=10)
    plt.tight_layout()
    png_path = os.path.join(IMPORTANCE_DIR, f"{run_id}.png")
    plt.savefig(png_path)
    plt.close(fig)
    print(f"  saved {csv_path} and {png_path}")


# load and split same as final_main
df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
train_df, val_df, test_df = data_util.split_by_prompt(df)
for dfx in (train_df, val_df, test_df):
    dfx['label'] = rename_yi_large(dfx['label'].to_numpy())

y_train = train_df['label'].to_numpy()
y_val = val_df['label'].to_numpy()
y_combined = np.concatenate([y_train, y_val])

# pick winners from sweep
sweep = pd.read_csv("final_results/validation_sweep/sweep_results.csv")
winners = {}
for feature_set in ['lexical', 'stylometric']:
    for classifier_name in ['RandomForest', 'LogReg', 'GaussianNB']:
        subset = sweep[(sweep['feature_set'] == feature_set) & (sweep['classifier'] == classifier_name)]
        winners[(feature_set, classifier_name)] = subset.loc[subset['val_macro_f1'].idxmax()]


# load train and val cached features for a winner
def load_combined(feature_set, winner_row):
    if feature_set == 'lexical':
        ngram_str = winner_row['ngram_range']
        use_tfidf = winner_row['use_tfidf']
        tsvd_rank = int(winner_row['tsvd_rank'])
        base = os.path.join(DATA_DIR, f"X_lex_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}")
        X_train = np.load(f"{base}_train.npy")
        X_val = np.load(f"{base}_val.npy")
        feature_names = [f"svd_{i}" for i in range(tsvd_rank)]
    else:
        X_train = np.load(os.path.join(DATA_DIR, "X_stylometric_train.npy"), allow_pickle=True)
        X_val = np.load(os.path.join(DATA_DIR, "X_stylometric_val.npy"), allow_pickle=True)
        feature_names = STYLOMETRIC_FEATURE_NAMES
    return np.concatenate([X_train, X_val], axis=0), feature_names


# rf gini and surrogate tree generaotion
for feature_set in ['lexical', 'stylometric']:
    winner_row = winners[(feature_set, 'RandomForest')]
    print(f"refitting {feature_set} rf winner: {winner_row['run_id']}")
    X_combined, feature_names = load_combined(feature_set, winner_row)

    classifier = make_rf(winner_row['classifier_hyperparameter_value'])
    classifier.train(X_combined, y_combined)

    top_n = 30 if feature_set == 'lexical' else len(feature_names)
    save_importance(winner_row['run_id'], feature_names, classifier.model.feature_importances_, top_n, "gini importance")

    # also dump surrogate tree
    classifier.surrogate_tree(X_combined,
        feature_names=feature_names, class_names=sorted(set(y_combined)),
        max_depth=3,
        save_name=os.path.join(SURROGATE_DIR, f"{winner_row['run_id']}.png"))
    print(f"  saved surrogate tree")


# logreg ovr coef magnitudes
for feature_set in ['lexical', 'stylometric']:
    winner_row = winners[(feature_set, 'LogReg')]
    print(f"refitting {feature_set} logreg winner as ovr: {winner_row['run_id']}")
    X_combined, feature_names = load_combined(feature_set, winner_row)

    ovr = OneVsRestClassifier(LogisticRegression(
        C=float(winner_row['classifier_hyperparameter_value']),
        max_iter=1000, random_state=42))
    ovr.fit(X_combined, y_combined)

    per_class_coefs = np.vstack([est.coef_ for est in ovr.estimators_])
    coef_magnitude = np.abs(per_class_coefs).sum(axis=0)

    top_n = 30 if feature_set == 'lexical' else len(feature_names)
    save_importance(winner_row['run_id'], feature_names, coef_magnitude, top_n, "logreg coef magnitude")


# gnb log-likelihood evidence
for feature_set in ['lexical', 'stylometric']:
    winner_row = winners[(feature_set, 'GaussianNB')]
    print(f"refitting {feature_set} gnb winner: {winner_row['run_id']}")
    X_combined, feature_names = load_combined(feature_set, winner_row)

    classifier = GaussianNB(var_smoothing=float(winner_row['classifier_hyperparameter_value']))
    classifier.fit(X_combined, y_combined)

    # grab each samples true-class gaussian params
    class_to_index = {c: i for i, c in enumerate(classifier.classes_)}
    class_indices = np.array([class_to_index[c] for c in y_combined])
    class_means = classifier.theta_[class_indices]
    class_variances = classifier.var_[class_indices]

    log_likelihood = -0.5 * np.log(2 * np.pi * class_variances) - (X_combined - class_means)**2 / (2 * class_variances)
    evidence = log_likelihood.sum(axis=0)

    top_n = 30 if feature_set == 'lexical' else len(feature_names)
    save_importance(winner_row['run_id'], feature_names, evidence, top_n, "gnb log-likelihood evidence")

print("done")
