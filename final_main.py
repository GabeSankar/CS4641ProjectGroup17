import os
import time
import numpy as np
import pandas as pd
from itertools import product

import data_util
import models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


# prep
DATA_DIR = "final_data"
SWEEP_DIR = "final_results/validation_sweep"
TRAIN_PRED_DIR = os.path.join(SWEEP_DIR, "predictions", "train")
VAL_PRED_DIR = os.path.join(SWEEP_DIR, "predictions", "val")

os.makedirs(TRAIN_PRED_DIR, exist_ok=True)
os.makedirs(VAL_PRED_DIR, exist_ok=True)

# lex feature grid 18 configs total
ngram_ranges = [(1,1), (1,2), (2,2)]
tfidf_flags = [False, True]
ranks = [50, 100, 300]

# classifier subgrid 17 configs total, per feature config
rf_max_features = ['sqrt', 0.25, 0.5, 0.75, 1.0]
lr_C = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
nb_var_smoothing = [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]

# sty feature names, ported from main.py
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


# cm to wide rows for csv
def cm_to_rows(run_id, split, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, t in enumerate(labels):
        row = {"run_id": run_id, "split": split, "true_label": t}
        for j, p in enumerate(labels):
            row[p] = int(cm[i, j])
        rows.append(row)
    return rows


# rf max_features can be sqrt or a float so we try float fallback string
def make_classifier(classifier_name, hyperparameter_value):
    if classifier_name == 'RandomForest':
        try:
            v = float(hyperparameter_value)
        except ValueError:
            v = hyperparameter_value
        return models.RandomForestClassifierWrapper(max_features=v)
    if classifier_name == 'LogReg':
        return models.LogisticRegressionWrapper(C=float(hyperparameter_value), max_iter=1000)
    return models.NaiveBayesClassifierWrapper(var_smoothing=float(hyperparameter_value))


# fix labels make sure it all lines up so same ordering for metrics and predictions
df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")
train_df, val_df, test_df = data_util.split_by_prompt(df)
for dfx in (train_df, val_df, test_df):
    dfx['label'] = rename_yi_large(dfx['label'].to_numpy())

print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

y_train = train_df['label'].to_numpy()
y_val = val_df['label'].to_numpy()
all_labels = sorted(set(y_train) | set(y_val) | set(test_df['label'].to_numpy()))

# stylometric is fixed, load once
X_sty_train = np.load(os.path.join(DATA_DIR, "X_stylometric_train.npy"), allow_pickle=True)
X_sty_val = np.load(os.path.join(DATA_DIR, "X_stylometric_val.npy"), allow_pickle=True)

metrics_rows = []
cm_rows = []


# score, record metrics, save preds
def record_run(run_id, feature_set, config_cols, train_preds, val_preds, fit_time):
    row = {"run_id": run_id, "feature_set": feature_set,
           "alpha": np.nan, "lex_run_id": np.nan, "sty_run_id": np.nan,
           **config_cols,
           "train_accuracy": accuracy_score(y_train, train_preds),
           "train_macro_f1": f1_score(y_train, train_preds, average='macro'),
           "val_accuracy": accuracy_score(y_val, val_preds),
           "val_macro_f1": f1_score(y_val, val_preds, average='macro'),
           "fit_time_sec": fit_time}
    metrics_rows.append(row)
    cm_rows.extend(cm_to_rows(run_id, "train", y_train, train_preds, all_labels))
    cm_rows.extend(cm_to_rows(run_id, "val", y_val, val_preds, all_labels))
    np.save(os.path.join(TRAIN_PRED_DIR, f"{run_id}.npy"), train_preds)
    np.save(os.path.join(VAL_PRED_DIR, f"{run_id}.npy"), val_preds)


# lex sweep per each classifier config
for ngram_range, use_tfidf, tsvd_rank in product(ngram_ranges, tfidf_flags, ranks):
    ngram_str = f"{ngram_range[0]}-{ngram_range[1]}"
    print(f"lex ngram={ngram_str} tfidf={use_tfidf} rank={tsvd_rank}")
    base = os.path.join(DATA_DIR, f"X_lex_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}")
    X_train = np.load(f"{base}_train.npy")
    X_val = np.load(f"{base}_val.npy")
    config = {"ngram_range": ngram_str, "use_tfidf": use_tfidf, "tsvd_rank": tsvd_rank}

    for v in rf_max_features:
        run_id = f"lexical_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}_RandomForest_max_features={v}"
        classifier = models.RandomForestClassifierWrapper(max_features=v)
        start_time = time.perf_counter()
        classifier.train(X_train, y_train)
        fit_time = time.perf_counter() - start_time
        record_run(run_id, "lexical", {**config, "classifier": "RandomForest", "classifier_hyperparameter_name": "max_features", "classifier_hyperparameter_value": str(v)}, classifier.predict(X_train), classifier.predict(X_val), fit_time)

    for v in lr_C:
        run_id = f"lexical_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}_LogReg_C={v}"
        classifier = models.LogisticRegressionWrapper(C=v, max_iter=1000)
        start_time = time.perf_counter()
        classifier.train(X_train, y_train)
        fit_time = time.perf_counter() - start_time
        record_run(run_id, "lexical", {**config, "classifier": "LogReg", "classifier_hyperparameter_name": "C", "classifier_hyperparameter_value": str(v)}, classifier.predict(X_train), classifier.predict(X_val), fit_time)

    for v in nb_var_smoothing:
        run_id = f"lexical_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}_GaussianNB_var_smoothing={v}"
        classifier = models.NaiveBayesClassifierWrapper(var_smoothing=v)
        start_time = time.perf_counter()
        classifier.train(X_train, y_train)
        fit_time = time.perf_counter() - start_time
        record_run(run_id, "lexical", {**config, "classifier": "GaussianNB", "classifier_hyperparameter_name": "var_smoothing", "classifier_hyperparameter_value": str(v)}, classifier.predict(X_train), classifier.predict(X_val), fit_time)

# stylometric sweep per classifier configs
print("stylometric one shot")
sty_config = {"ngram_range": np.nan, "use_tfidf": np.nan, "tsvd_rank": np.nan}

for v in rf_max_features:
    run_id = f"stylometric_RandomForest_max_features={v}"
    classifier = models.RandomForestClassifierWrapper(max_features=v)
    start_time = time.perf_counter()
    classifier.train(X_sty_train, y_train)
    fit_time = time.perf_counter() - start_time
    record_run(run_id, "stylometric", {**sty_config, "classifier": "RandomForest", "classifier_hyperparameter_name": "max_features", "classifier_hyperparameter_value": str(v)}, classifier.predict(X_sty_train), classifier.predict(X_sty_val), fit_time)

for v in lr_C:
    run_id = f"stylometric_LogReg_C={v}"
    classifier = models.LogisticRegressionWrapper(C=v, max_iter=1000)
    start_time = time.perf_counter()
    classifier.train(X_sty_train, y_train)
    fit_time = time.perf_counter() - start_time
    record_run(run_id, "stylometric", {**sty_config, "classifier": "LogReg", "classifier_hyperparameter_name": "C", "classifier_hyperparameter_value": str(v)}, classifier.predict(X_sty_train), classifier.predict(X_sty_val), fit_time)

for v in nb_var_smoothing:
    run_id = f"stylometric_GaussianNB_var_smoothing={v}"
    classifier = models.NaiveBayesClassifierWrapper(var_smoothing=v)
    start_time = time.perf_counter()
    classifier.train(X_sty_train, y_train)
    fit_time = time.perf_counter() - start_time
    record_run(run_id, "stylometric", {**sty_config, "classifier": "GaussianNB", "classifier_hyperparameter_name": "var_smoothing", "classifier_hyperparameter_value": str(v)}, classifier.predict(X_sty_train), classifier.predict(X_sty_val), fit_time)

# hybrid sweep across 0 to 1 alpha range
hybrid_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
metrics_df = pd.DataFrame(metrics_rows)

for classifier_name in ['RandomForest', 'LogReg', 'GaussianNB']:
    lex_subset = metrics_df[(metrics_df['feature_set'] == 'lexical') & (metrics_df['classifier'] == classifier_name)]
    sty_subset = metrics_df[(metrics_df['feature_set'] == 'stylometric') & (metrics_df['classifier'] == classifier_name)]
    best_lex = lex_subset.loc[lex_subset['val_macro_f1'].idxmax()]
    best_sty = sty_subset.loc[sty_subset['val_macro_f1'].idxmax()]
    print(f"hybrid {classifier_name}: best lex {best_lex['run_id']}, best sty {best_sty['run_id']}")

    # load best lex features
    ngram_str = best_lex['ngram_range']
    use_tfidf = best_lex['use_tfidf']
    tsvd_rank = int(best_lex['tsvd_rank'])
    base = os.path.join(DATA_DIR, f"X_lex_ngram={ngram_str}_tfidf={use_tfidf}_rank={tsvd_rank}")
    X_lex_train = np.load(f"{base}_train.npy")
    X_lex_val = np.load(f"{base}_val.npy")

    # rebuild best lex and sty classifiers
    lex_classifier = make_classifier(classifier_name, best_lex['classifier_hyperparameter_value'])
    sty_classifier = make_classifier(classifier_name, best_sty['classifier_hyperparameter_value'])

    lex_classifier.train(X_lex_train, y_train)
    sty_classifier.train(X_sty_train, y_train)

    lex_proba_train = lex_classifier.model.predict_proba(X_lex_train)
    lex_proba_val = lex_classifier.model.predict_proba(X_lex_val)
    sty_proba_train = sty_classifier.model.predict_proba(X_sty_train)
    sty_proba_val = sty_classifier.model.predict_proba(X_sty_val)
    classes = lex_classifier.model.classes_

    for alpha in hybrid_alphas:
        train_preds = classes[np.argmax(alpha * sty_proba_train + (1 - alpha) * lex_proba_train, axis=1)]
        val_preds = classes[np.argmax(alpha * sty_proba_val + (1 - alpha) * lex_proba_val, axis=1)]
        run_id = f"hybrid_{classifier_name}_alpha={alpha}"
        config = {"ngram_range": ngram_str, "use_tfidf": use_tfidf, "tsvd_rank": tsvd_rank,
                  "classifier": classifier_name, "classifier_hyperparameter_name": "alpha", "classifier_hyperparameter_value": str(alpha),
                  "alpha": alpha, "lex_run_id": best_lex['run_id'], "sty_run_id": best_sty['run_id']}
        record_run(run_id, "hybrid", config, train_preds, val_preds, 0.0)

# pick best lex/sty/hybrid per classifier + best overall
TEST_DIR = "final_results/testing_winners"
TEST_PRED_DIR = os.path.join(TEST_DIR, "predictions")
PLOT_DIR = os.path.join(TEST_DIR, "confusion_matrix_plots")
SURROGATE_DIR = os.path.join(TEST_DIR, "surrogate_trees")
os.makedirs(TEST_PRED_DIR, exist_ok=True)
os.makedirs(SURROGATE_DIR, exist_ok=True)
for sub in ['lexical', 'stylometric', 'hybrid']:
    os.makedirs(os.path.join(PLOT_DIR, sub), exist_ok=True)
    os.makedirs(os.path.join(TEST_PRED_DIR, sub), exist_ok=True)

train_texts = train_df['text'].tolist()
val_texts = val_df['text'].tolist()
test_texts = test_df['text'].tolist()
combined_texts = train_texts + val_texts
X_sty_test = np.load(os.path.join(DATA_DIR, "X_stylometric_test.npy"), allow_pickle=True)
y_test = test_df['label'].to_numpy()
y_combined = np.concatenate([y_train, y_val])
X_sty_combined = np.concatenate([X_sty_train, X_sty_val], axis=0)

metrics_df = pd.DataFrame(metrics_rows)
winners = []
for feature_set in ['lexical', 'stylometric', 'hybrid']:
    for classifier_name in ['RandomForest', 'LogReg', 'GaussianNB']:
        subset = metrics_df[(metrics_df['feature_set'] == feature_set) & (metrics_df['classifier'] == classifier_name)]
        winners.append((f"{feature_set}_{classifier_name}", subset.loc[subset['val_macro_f1'].idxmax()]))
winners.append(('overall_winner', metrics_df.loc[metrics_df['val_macro_f1'].idxmax()]))


# human readable winner label
def human_label(label):
    if label == 'overall_winner':
        return 'Overall Winner'
    feature_set, classifier_name = label.split('_', 1)
    return f"Best {feature_set.capitalize()} {classifier_name}"


# refit each winner on train+val and score on test once
test_metrics_rows = []
test_cm_rows = []
winner_blocks = []
for label, w in winners:
    classifier_name = w['classifier']
    feature_set = w['feature_set']
    print(f"test eval {label}")

    if feature_set == 'lexical':
        ngram_range = tuple(int(x) for x in w['ngram_range'].split('-'))
        use_tfidf = bool(w['use_tfidf'])
        tsvd_rank = int(w['tsvd_rank'])
        vectorizer, svd, X_combined = data_util.build_lex_pipeline(combined_texts, ngram_range, use_tfidf, tsvd_rank)
        X_test_lex = svd.transform(vectorizer.transform(test_texts))
        classifier = make_classifier(classifier_name, w['classifier_hyperparameter_value'])
        classifier.train(X_combined, y_combined)
        test_preds = classifier.predict(X_test_lex)
        # also dump surrogate tree for rf
        if classifier_name == 'RandomForest' and label != 'overall_winner':
            classifier.surrogate_tree(X_combined,
                feature_names=[f"svd_{i}" for i in range(tsvd_rank)],
                class_names=all_labels, max_depth=3,
                save_name=os.path.join(SURROGATE_DIR, f"{w['run_id']}.png"))
    elif feature_set == 'stylometric':
        classifier = make_classifier(classifier_name, w['classifier_hyperparameter_value'])
        classifier.train(X_sty_combined, y_combined)
        test_preds = classifier.predict(X_sty_test)
        if classifier_name == 'RandomForest' and label != 'overall_winner':
            # also dump surrogate tree for rf
            classifier.surrogate_tree(X_sty_combined,
                feature_names=STYLOMETRIC_FEATURE_NAMES,
                class_names=all_labels, max_depth=3,
                save_name=os.path.join(SURROGATE_DIR, f"{w['run_id']}.png"))
    else:
        ngram_range = tuple(int(x) for x in w['ngram_range'].split('-'))
        use_tfidf = bool(w['use_tfidf'])
        tsvd_rank = int(w['tsvd_rank'])
        lex_winner_row = metrics_df[metrics_df['run_id'] == w['lex_run_id']].iloc[0]
        sty_winner_row = metrics_df[metrics_df['run_id'] == w['sty_run_id']].iloc[0]
        vectorizer, svd, X_lex_combined = data_util.build_lex_pipeline(combined_texts, ngram_range, use_tfidf, tsvd_rank)
        X_lex_test = svd.transform(vectorizer.transform(test_texts))
        lex_classifier = make_classifier(classifier_name, lex_winner_row['classifier_hyperparameter_value'])
        lex_classifier.train(X_lex_combined, y_combined)
        sty_classifier = make_classifier(classifier_name, sty_winner_row['classifier_hyperparameter_value'])
        sty_classifier.train(X_sty_combined, y_combined)
        lex_proba = lex_classifier.model.predict_proba(X_lex_test)
        sty_proba = sty_classifier.model.predict_proba(X_sty_test)
        alpha = w['alpha']
        classes = lex_classifier.model.classes_
        test_preds = classes[np.argmax(alpha * sty_proba + (1 - alpha) * lex_proba, axis=1)]

    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='macro')
    per_class = f1_score(y_test, test_preds, average=None, labels=all_labels)
    row = {"winner": label, "run_id": w['run_id'], "feature_set": feature_set,
           "ngram_range": w['ngram_range'], "use_tfidf": w['use_tfidf'], "tsvd_rank": w['tsvd_rank'],
           "classifier": classifier_name, "classifier_hyperparameter_name": w['classifier_hyperparameter_name'],
           "classifier_hyperparameter_value": w['classifier_hyperparameter_value'],
           "alpha": w['alpha'], "lex_run_id": w['lex_run_id'], "sty_run_id": w['sty_run_id'],
           "val_macro_f1": w['val_macro_f1'],
           "test_accuracy": test_acc, "test_macro_f1": test_f1}
    for c, class_f1 in zip(all_labels, per_class):
        row[f"{c}_f1"] = class_f1
    test_metrics_rows.append(row)
    test_cm_rows.extend(cm_to_rows(w['run_id'], 'test', y_test, test_preds, all_labels))
    if label == 'overall_winner':
        np.save(os.path.join(TEST_PRED_DIR, f"overall_winner_{w['run_id']}.npy"), test_preds)
    else:
        np.save(os.path.join(TEST_PRED_DIR, feature_set, f"{w['run_id']}.npy"), test_preds)

    # plot png nd text grid, filenames carry the run_id
    run_id = w['run_id']
    if label == 'overall_winner':
        png_path = os.path.join(PLOT_DIR, f"overall_winner_{run_id}.png")
        txt_path = os.path.join(PLOT_DIR, f"overall_winner_{run_id}.txt")
    else:
        png_path = os.path.join(PLOT_DIR, feature_set, f"{run_id}.png")
        txt_path = os.path.join(PLOT_DIR, feature_set, f"{run_id}.txt")
    cm = confusion_matrix(y_test, test_preds, labels=all_labels)
    title = f"{human_label(label)} | {run_id}"
    fig, ax = plt.subplots(figsize=(10, 7))
    ConfusionMatrixDisplay(cm, display_labels=all_labels).plot(ax=ax, xticks_rotation=45)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    # cm text grid used in txt and summary
    cm_lines = ["true\\pred".ljust(15) + " ".join(c.ljust(12) for c in all_labels)]
    for i, c in enumerate(all_labels):
        cm_lines.append(c.ljust(15) + " ".join(str(int(cm[i, j])).ljust(12) for j in range(len(all_labels))))
    cm_text = "\n".join(cm_lines)
    with open(txt_path, 'w') as f:
        f.write(cm_text + "\n")

    # summary block for this winner
    block = []
    bar = "=" * 60
    block.append(bar)
    block.append(f"WINNER: {human_label(label)}")
    block.append(bar)
    block.append(f"Run ID: {w['run_id']}")
    block.append("")
    block.append("Configuration:")
    block.append(f"  feature_set:    {feature_set}")
    if feature_set == 'hybrid':
        lex_winner_row = metrics_df[metrics_df['run_id'] == w['lex_run_id']].iloc[0]
        sty_winner_row = metrics_df[metrics_df['run_id'] == w['sty_run_id']].iloc[0]
        block.append(f"  classifier:     {classifier_name}")
        block.append(f"  alpha:          {w['alpha']}")
        block.append("")
        block.append("  Lex side:")
        block.append(f"    run_id:       {lex_winner_row['run_id']}")
        block.append(f"    ngram_range:  {lex_winner_row['ngram_range']}")
        block.append(f"    use_tfidf:    {lex_winner_row['use_tfidf']}")
        block.append(f"    tsvd_rank:    {int(lex_winner_row['tsvd_rank'])}")
        block.append(f"    {lex_winner_row['classifier_hyperparameter_name']}: {lex_winner_row['classifier_hyperparameter_value']}")
        block.append("")
        block.append("  Sty side:")
        block.append(f"    run_id:       {sty_winner_row['run_id']}")
        block.append(f"    {sty_winner_row['classifier_hyperparameter_name']}: {sty_winner_row['classifier_hyperparameter_value']}")
    elif feature_set == 'lexical':
        block.append(f"  ngram_range:    {w['ngram_range']}")
        block.append(f"  use_tfidf:      {w['use_tfidf']}")
        block.append(f"  tsvd_rank:      {int(w['tsvd_rank'])}")
        block.append(f"  classifier:     {classifier_name}")
        block.append(f"  {w['classifier_hyperparameter_name']+':':<16}{w['classifier_hyperparameter_value']}")
    else:
        block.append(f"  classifier:     {classifier_name}")
        block.append(f"  {w['classifier_hyperparameter_name']+':':<16}{w['classifier_hyperparameter_value']}")
    block.append("")
    block.append("Validation:")
    block.append(f"  accuracy:       {w['val_accuracy']:.4f}")
    block.append(f"  macro_f1:       {w['val_macro_f1']:.4f}")
    block.append("")
    block.append("Test Set Evaluation:")
    block.append(f"  accuracy:       {test_acc:.4f}")
    block.append(f"  macro_f1:       {test_f1:.4f}")
    block.append("")
    # per class f1
    block.append("Per-class F1:")
    for c, class_f1 in zip(all_labels, per_class):
        block.append(f"  {c:<15} {class_f1:.4f}")
    block.append("")
    # classification report
    block.append("Classification Report:")
    block.append(classification_report(y_test, test_preds, labels=all_labels, digits=4))
    # confusion matrix
    block.append("Confusion Matrix:")
    block.append(cm_text)
    block.append("")
    winner_blocks.append("\n".join(block))

# save validation sweep csvs
pd.DataFrame(metrics_rows).to_csv(os.path.join(SWEEP_DIR, "sweep_results.csv"), index=False)
pd.DataFrame(cm_rows).to_csv(os.path.join(SWEEP_DIR, "confusion_matrices.csv"), index=False)

# save testing winners csvs and summary
pd.DataFrame(test_metrics_rows).to_csv(os.path.join(TEST_DIR, "test_results.csv"), index=False)
pd.DataFrame(test_cm_rows).to_csv(os.path.join(TEST_DIR, "confusion_matrices.csv"), index=False)
with open(os.path.join(TEST_DIR, "summary.txt"), 'w') as f:
    f.write("\n".join(winner_blocks))

print("all results saved to final_results/")
