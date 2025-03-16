import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import logging
logging.set_verbosity_error()

import random
import numpy as np
import pandas as pd
import tensorflow as tf

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import mlflow
import mlflow.tensorflow
from datetime import datetime
from transformers import AutoTokenizer, TFBertModel
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping  # Removed TensorBoard
from tensorflow.keras import mixed_precision
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet

from calculate_FeReRe_Metrics import compute_metrics

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(text: str) -> str:
    """Basic stopword removal."""
    words = word_tokenize(text)
    filtered_words = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(filtered_words)

def get_synonym(word):
    """Return a random synonym from WordNet (if available)."""
    synsets = wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != word.lower():
                return lemma_name
    return None

def augment_data_with_synonyms(text: str, replacement_prob=0.15) -> str:
    """Randomly replaces some words with synonyms."""
    words = word_tokenize(text)
    augmented_words = []
    for w in words:
        if random.random() < replacement_prob and w.isalpha():
            syn = get_synonym(w)
            if syn:
                augmented_words.append(syn)
            else:
                augmented_words.append(w)
        else:
            augmented_words.append(w)
    return " ".join(augmented_words)

def split_sentences(text: str) -> str:
    """Split the text into sentences, re-join them with '||'."""
    sents = sent_tokenize(text)
    return " || ".join(sents)

def load_and_combine_excel(
    file_list,
    axis=0,
    remove_sw=False,
    augment=False,
    do_split=False,
    header=None
):
    """
    Loads multiple Excel files and combines them (by rows or columns).
    If remove_sw/augment/do_split are true, apply those transformations to col=1 (the text column).
    """
    combined_df = None
    for idx, fpath in enumerate(file_list):
        df = pd.read_excel(fpath, header=header)
        if df.shape[1] > 1:  # We have an ID column + text column
            if remove_sw or augment or do_split:
                processed_texts = []
                for txt in df.iloc[:, 1]:
                    if isinstance(txt, str):
                        if remove_sw:
                            txt = remove_stopwords(txt)
                        if augment:
                            txt = augment_data_with_synonyms(txt)
                        if do_split:
                            txt = split_sentences(txt)
                    processed_texts.append(txt)
                df.iloc[:, 1] = processed_texts

        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], axis=axis, ignore_index=True)

    return combined_df

def train_eval_kfold_combined(
    feedback_files,
    requirements_files,
    ground_truth_files,
    n_splits=5,
    epochs=30,
    batch_size=32,
    max_length=256,
    model_name="bert-base-uncased",
    remove_sw=False,
    augment=False,
    do_split=False,
    random_state=42,
    log_dir="./logs_kfold",
    results_dir="./new_kfold_results",
    exp_name="FeReRe_KFold_Combined",
    exp_id=1
):
    """
    1) Load & combine feedback, requirements, ground-truth
    2) Create positive (req->fb) pairs from ground truth; create negative pairs
    3) Train/Eval with StratifiedKFold
    4) Save classification results and compute FeReRe metrics
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ============= 1. LOAD & COMBINE DATA =============
    feedback_df = load_and_combine_excel(
        feedback_files,
        axis=0,
        remove_sw=remove_sw,
        augment=augment,
        do_split=do_split,
        header=None
    )
    requirements_df = load_and_combine_excel(
        requirements_files,
        axis=0,
        remove_sw=remove_sw,
        augment=augment,
        do_split=do_split,
        header=None
    )
    # Ground-truth is wide-format => combine along columns
    ground_truth_df = load_and_combine_excel(
        ground_truth_files,
        axis=1,
        remove_sw=False,
        augment=False,
        do_split=False,
        header=None
    )

    # Build a dict: { requirement_id : list of feedback_ids }
    ground_truth_map = {}
    for col in ground_truth_df.columns:
        req_id = ground_truth_df[col].iloc[0]
        fb_list = ground_truth_df[col].iloc[1:].dropna().tolist()
        ground_truth_map[req_id] = fb_list

    # Prepare the actual text for feedback & requirements
    feedback_texts = feedback_df.iloc[:, 1].tolist()  # text in column 1
    requirements_texts = requirements_df.iloc[:, 1].tolist()

    # We'll do NLTK sentence tokenization on each
    feedback_sents = [sent_tokenize(t) for t in feedback_texts]
    requirements_sents = [sent_tokenize(t) for t in requirements_texts]

    # ============= 2. CREATE POSITIVE & NEGATIVE SAMPLES =============
    positive_samples = []
    positive_pairs = []  # (req_idx, fb_idx)
    for req_id, fb_ids in ground_truth_map.items():
        # find the row index for this requirement
        req_idx_list = requirements_df[requirements_df.iloc[:, 0] == req_id].index.tolist()
        if not req_idx_list:
            continue
        req_index = req_idx_list[0]
        req_s_list = requirements_sents[req_index]
        for fb_id in fb_ids:
            fb_idx_list = feedback_df[feedback_df.iloc[:, 0] == fb_id].index.tolist()
            if not fb_idx_list:
                continue
            fb_index = fb_idx_list[0]
            fb_s_list = feedback_sents[fb_index]
            for rs in req_s_list:
                for fs in fb_s_list:
                    positive_samples.append((rs, fs))
                    positive_pairs.append((req_index, fb_index))

    # Make negative samples
    negative_samples = []
    negative_pairs = []
    random.seed(random_state)
    while len(negative_samples) < len(positive_samples):
        ridx = random.choice(requirements_df.index)
        fidx = random.choice(feedback_df.index)
        if (ridx, fidx) not in positive_pairs:
            req_s_list = requirements_sents[ridx]
            fb_s_list = feedback_sents[fidx]
            for rs in req_s_list:
                for fs in fb_s_list:
                    negative_samples.append((rs, fs))
                    negative_pairs.append((ridx, fidx))
                    if len(negative_samples) >= len(positive_samples):
                        break
                if len(negative_samples) >= len(positive_samples):
                    break

    all_samples = positive_samples + negative_samples
    all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    all_pairs = positive_pairs + negative_pairs

    # ============= 3. TOKENIZE AND STRATIFIED K-FOLD =============
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        [p[0] for p in all_samples],  # requirement sentence
        [p[1] for p in all_samples],  # feedback sentence
        return_tensors='np',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_fold_metrics = []
    current_fold = 1

    mlflow.set_experiment(exp_name)
    mlflow.tensorflow.autolog(disable=True)

    with mlflow.start_run(run_name=f"KFold_Exp_{exp_id}"):
        # Log hyperparams
        mlflow.log_param("exp_id", exp_id)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("remove_sw", remove_sw)
        mlflow.log_param("augment", augment)
        mlflow.log_param("do_split", do_split)

        # Mixed precision
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        for train_index, test_index in skf.split(input_ids, all_labels):
            print(f"\n===== [FOLD {current_fold}] =====")
            X_train_ids = input_ids[train_index]
            X_test_ids  = input_ids[test_index]
            X_train_mask = attention_mask[train_index]
            X_test_mask  = attention_mask[test_index]
            y_train = np.array(all_labels)[train_index]
            y_test  = np.array(all_labels)[test_index]
            test_pairs = [all_pairs[i] for i in test_index]

            # Build model
            bert_model = TFBertModel.from_pretrained(model_name)
            in_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
            in_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

            outputs = bert_model(in_ids, attention_mask=in_mask)
            pooled_output = outputs.pooler_output
            final_output = tf.keras.layers.Dense(1, activation="sigmoid")(pooled_output)
            model = tf.keras.Model(inputs=[in_ids, in_mask], outputs=final_output)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Callbacks: using only EarlyStopping
            early_stop = EarlyStopping(patience=4, monitor="val_loss", restore_best_weights=True)

            # Train without TensorBoard callback
            history = model.fit(
                [X_train_ids, X_train_mask],
                y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=1
            )

            # Predict on test fold
            preds = model.predict([X_test_ids, X_test_mask])
            preds_label = (preds > 0.5).astype("int32").flatten()

            # Convert predictions into wide-format {req_id: [fb_ids]}
            classified_dict = {}
            for rid in requirements_df.iloc[:, 0]:
                classified_dict[rid] = []

            for idx, label_val in zip(test_index, preds_label):
                if label_val == 1:
                    (req_index, fb_index) = all_pairs[idx]
                    req_id = requirements_df.iloc[req_index, 0]
                    fb_id  = feedback_df.iloc[fb_index, 0]
                    if fb_id not in classified_dict[req_id]:
                        classified_dict[req_id].append(fb_id)

            # Save classification results
            max_fb_len = max(len(v) for v in classified_dict.values()) if classified_dict else 0
            for rid in classified_dict:
                classified_dict[rid] += [''] * (max_fb_len - len(classified_dict[rid]))
            df_classified = pd.DataFrame(classified_dict)
            fold_classified_path = os.path.join(results_dir, f"classified_feedback_requirements_exp_{exp_id}_fold_{current_fold}.xlsx")
            df_classified.to_excel(fold_classified_path, index=False)

            # Build wide-format ground truth for the test fold only
            test_gt_dict = {}
            for rid in requirements_df.iloc[:, 0]:
                test_gt_dict[rid] = []
            for i in test_index:
                req_idx, fb_idx = all_pairs[i]
                rid = requirements_df.iloc[req_idx, 0]
                fid = feedback_df.iloc[fb_idx, 0]
                if fid in ground_truth_map.get(rid, []):
                    test_gt_dict[rid].append(fid)

            max_gt_len = max(len(v) for v in test_gt_dict.values()) if test_gt_dict else 0
            for rid in test_gt_dict:
                test_gt_dict[rid] += [''] * (max_gt_len - len(test_gt_dict[rid]))
            df_test_gt = pd.DataFrame(test_gt_dict)
            fold_gt_path = os.path.join(results_dir, f"test_ground_truth_exp_{exp_id}_fold_{current_fold}.xlsx")
            df_test_gt.to_excel(fold_gt_path, index=False)

            # ========== 4. Compute FeReRe metrics ==========
            classifier_df = pd.read_excel(fold_classified_path, header=None)
            ground_truth_f = pd.read_excel(fold_gt_path, header=None)
            p, r, f2, avg_assign = compute_metrics(classifier_df, ground_truth_f)

            print(f"[FOLD {current_fold}] => Precision={p:.4f}, Recall={r:.4f}, F2={f2:.4f}, AvgAssigned={avg_assign:.2f}")
            fold_res = {
                "fold": current_fold,
                "precision": p,
                "recall": r,
                "f2": f2,
                "avg_assign": avg_assign
            }
            all_fold_metrics.append(fold_res)

            # Log fold metrics
            mlflow.log_metric(f"fold_{current_fold}_precision", p)
            mlflow.log_metric(f"fold_{current_fold}_recall", r)
            mlflow.log_metric(f"fold_{current_fold}_f2", f2)
            mlflow.log_metric(f"fold_{current_fold}_avg_assigned", avg_assign)
            
            # Clean up
            del model
            del bert_model
            tf.keras.backend.clear_session()
            current_fold += 1

        # Summarize across folds
        if len(all_fold_metrics) > 0:
            avg_p = np.mean([m["precision"] for m in all_fold_metrics])
            avg_r = np.mean([m["recall"] for m in all_fold_metrics])
            avg_f2 = np.mean([m["f2"] for m in all_fold_metrics])
            avg_as = np.mean([m["avg_assign"] for m in all_fold_metrics])

            print("\n===== OVERALL K-FOLD RESULTS =====")
            print(f"Precision={avg_p:.4f}, Recall={avg_r:.4f}, F2={avg_f2:.4f}, AvgAssigned={avg_as:.2f}")

            mlflow.log_metric("avg_precision", avg_p)
            mlflow.log_metric("avg_recall", avg_r)
            mlflow.log_metric("avg_f2", avg_f2)
            mlflow.log_metric("avg_assigned", avg_as)

    return all_fold_metrics
