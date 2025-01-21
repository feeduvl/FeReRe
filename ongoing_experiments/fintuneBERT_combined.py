
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use only the first GPU (index 3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import logging
logging.set_verbosity_error()

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from datetime import datetime
from transformers import AutoTokenizer, TFAutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import mixed_precision

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet

from calculate_FeReRe_Metrics import compute_metrics


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Global stopwords set
STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(text: str) -> str:
    """Removes stopwords from the input text."""
    words = word_tokenize(text)
    filtered_words = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(filtered_words)

def get_synonym(word):
    """Returns a synonym for the given word (if available) from WordNet."""
    synsets = wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != word.lower():
                return lemma_name
    return None

def augment_data_with_synonyms(text: str, replacement_prob=0.15) -> str:
    """Augments text by randomly replacing words with their synonyms."""
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
    """Splits text into sentences and joins them with ' || '."""
    sents = sent_tokenize(text)
    return " || ".join(sents)

def apply_preprocessing(feedback_file, requirements_file, gt_file,
                        remove_sw=False, augment=False, split=False):
    """
    Reads the Excel files for feedback, requirements, and ground truth.
    Applies the chosen preprocessing steps (stopword removal, synonym augmentation, sentence splitting)
    to the text columns (assumed to be column index 1) of both feedback and requirements.
    The ground truth file is left unchanged.
    Saves the processed files into a temporary folder and returns the new file paths.
    """
    feedback_df = pd.read_excel(feedback_file, header=None)
    requirements_df = pd.read_excel(requirements_file, header=None)
    gt_df = pd.read_excel(gt_file)

    # Apply preprocessing only on the text column (index=1)
    for df in [feedback_df, requirements_df]:
        processed_texts = []
        for text in df.iloc[:, 1]:
            if isinstance(text, str):
                if remove_sw:
                    text = remove_stopwords(text)
                if augment:
                    text = augment_data_with_synonyms(text)
                if split:
                    text = split_sentences(text)
            processed_texts.append(text)
        df.iloc[:, 1] = processed_texts

    from pathlib import Path
    tmp_dir = Path("./ongoing_experiments/temp_experiments_single")
    tmp_dir.mkdir(exist_ok=True)

    suffix_parts = []
    if remove_sw: suffix_parts.append("sw")
    if augment: suffix_parts.append("aug")
    if split: suffix_parts.append("split")
    suffix = "_".join(suffix_parts) if suffix_parts else "original"

    feedback_path = tmp_dir / f"feedback_{suffix}.xlsx"
    requirements_path = tmp_dir / f"requirements_{suffix}.xlsx"
    gt_path = tmp_dir / f"gt_{suffix}.xlsx"

    feedback_df.to_excel(feedback_path, index=False, header=False)
    requirements_df.to_excel(requirements_path, index=False, header=False)
    gt_df.to_excel(gt_path, index=False)

    return str(feedback_path), str(requirements_path), str(gt_path)


# ------------------------------
# Load and Combine Data Function
# ------------------------------
def load_and_combine_excel(
    file_list,
    axis=0,
    remove_sw=False,
    augment=False,
    do_split=False,
    header=None
):
    """
    Reads multiple Excel files, optionally applies preprocessing (to column index 1) 
    using our unified apply_preprocessing logic (if desired), and concatenates them along the given axis.
    """
    combined_df = None
    for i, f in enumerate(file_list):
        df = pd.read_excel(f, header=header)
        # Preprocess text in column=1 if applicable
        if df.shape[1] > 1:
            df.iloc[:, 1] = df.iloc[:, 1].apply(
                lambda x: (
                    split_sentences(augment_data_with_synonyms(remove_stopwords(x)))
                    if isinstance(x, str) and (remove_sw or augment or do_split)
                    else x
                )
            ) if (remove_sw or augment or do_split) else df.iloc[:, 1]
        if i == 0:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], axis=axis, ignore_index=True)
    return combined_df


# =======================================
#  Main Single-run Combined Data Training
# =======================================
def train_eval_single_combined(
    feedback_files,
    requirements_files,
    ground_truth_files,
    epochs=30,
    batch_size=32,
    max_length=256,
    model_name="bert-base-uncased",
    remove_sw=False,
    augment=False,
    do_split=False,
    random_state=42,
    log_dir="./logs_single",      # new log directory name
    results_dir="./single_results",  # new results directory name
    exp_name="FeReRe_Single_Combined",  # new MLflow experiment name
    exp_id=1
):
    """
    1) Loads & merges feedback, requirements, and ground truth.
    2) Creates positive & negative samples.
    3) Splits the samples once into training and test sets.
    4) Logs the run with TensorBoard & MLflow.
    5) Saves the classified feedback and ground-truth files.
    6) Computes FeReRe metrics (Precision, Recall, F2, AvgAssigned).
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    print("[INFO] Loading & combining data (feedback, requirements, ground-truth)...")
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
    # ground_truth -> loaded along columns
    ground_truth_df = load_and_combine_excel(
        ground_truth_files,
        axis=1,
        remove_sw=False,  # IDs do not need text preprocessing
        augment=False,
        do_split=False,
        header=None
    )

    # 2. Build ground_truth dict { req_id : [fb_ids] }
    ground_truth_map = {}
    for col in ground_truth_df.columns:
        req_id = ground_truth_df[col].iloc[0]
        fb_list = ground_truth_df[col].iloc[1:].dropna().tolist()
        ground_truth_map[req_id] = fb_list

    # 3. Prepare text lists for feedback and requirements
    feedback_texts = feedback_df.iloc[:, 1].tolist()  # text in col=1
    requirements_texts = requirements_df.iloc[:, 1].tolist()

    # 4. Sentence-split them (use NLTK splitting)
    feedback_sents = [sent_tokenize(t) for t in feedback_texts]
    requirements_sents = [sent_tokenize(t) for t in requirements_texts]

    # 5. Create positive samples
    positive_samples = []
    positive_pairs = []
    for req_id, fb_ids in ground_truth_map.items():
        # find req_index
        req_idx_list = requirements_df[requirements_df.iloc[:, 0] == req_id].index.tolist()
        if len(req_idx_list) > 0:
            req_index = req_idx_list[0]
            req_s_list = requirements_sents[req_index]
            for fb_id in fb_ids:
                fb_idx_list = feedback_df[feedback_df.iloc[:, 0] == fb_id].index.tolist()
                if len(fb_idx_list) > 0:
                    fb_index = fb_idx_list[0]
                    fb_s_list = feedback_sents[fb_index]
                    for rs in req_s_list:
                        for fs in fb_s_list:
                            positive_samples.append((rs, fs))
                            positive_pairs.append((req_index, fb_index))

    # 6. Create negative samples (roughly same size as positives)
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

    # 7. Tokenize with BERT (or your chosen model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        [p[0] for p in all_samples],  # req sentence
        [p[1] for p in all_samples],  # fb sentence
        return_tensors='np',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # 8. Split into train and test sets (80%-20% stratified)
    from sklearn.model_selection import train_test_split
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test, train_pairs, test_pairs = train_test_split(
        input_ids, attention_mask, np.array(all_labels), all_pairs,
        stratify=all_labels, random_state=random_state, test_size=0.2
    )

    # Log experiment with MLflow (set new experiment name)
    mlflow.set_experiment(exp_name)
    mlflow.tensorflow.autolog(disable=True)

    with mlflow.start_run(run_name=f"SingleRun_Exp_{exp_id}"):
        mlflow.log_param("exp_id", exp_id)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("remove_sw", remove_sw)
        mlflow.log_param("augment", augment)
        mlflow.log_param("do_split", do_split)

        # Optionally enable mixed precision
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        # Build BERT model (or your chosen transformer) 
        bert_model = TFBertModel.from_pretrained(model_name)
        in_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
        outputs = bert_model(in_ids, attention_mask=in_mask)
        pooled_output = outputs.pooler_output  # shape: [batch, hidden_size]
        final_output = tf.keras.layers.Dense(1, activation="sigmoid")(pooled_output)
        model = tf.keras.Model(inputs=[in_ids, in_mask], outputs=final_output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Create TensorBoard log directory under the single run folder
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        fold_logdir = os.path.join(log_dir, f"Exp_{exp_id}_{now_str}")
        os.makedirs(fold_logdir, exist_ok=True)
        tb_callback = TensorBoard(log_dir=fold_logdir, histogram_freq=0, write_graph=True)
        early_stop = EarlyStopping(patience=4, monitor="val_loss", restore_best_weights=True)

        # Train the model
        history = model.fit(
            [X_train_ids, X_train_mask],
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tb_callback, early_stop],
            verbose=1
        )

        # Predict on the test set
        preds = model.predict([X_test_ids, X_test_mask])
        preds_label = (preds > 0.5).astype("int32").flatten()

        # Build wide-format "classified" results {req_id : [fb_ids]}
        classified_dict = {}
        for rid in requirements_df.iloc[:, 0]:
            classified_dict[rid] = []

        for idx, label_val in zip(range(len(test_pairs)), preds_label):
            if label_val == 1:
                (req_index, fb_index) = test_pairs[idx]
                req_id = requirements_df.iloc[req_index, 0]
                fb_id  = feedback_df.iloc[fb_index, 0]
                if fb_id not in classified_dict[req_id]:
                    classified_dict[req_id].append(fb_id)

        # Save the results to XLSX (naming changed for single run)
        max_fb_len = max(len(vals) for vals in classified_dict.values()) if classified_dict else 0
        for rid in classified_dict:
            classified_dict[rid] += [''] * (max_fb_len - len(classified_dict[rid]))
        df_classified = pd.DataFrame(classified_dict)
        single_classified_path = os.path.join(results_dir, f"classified_feedback_requirements_single_exp_{exp_id}.xlsx")
        df_classified.to_excel(single_classified_path, index=False)

        # Build wide-format ground truth for the test set
        test_gt_dict = {}
        for rid in requirements_df.iloc[:,0]:
            test_gt_dict[rid] = []
        for i in range(len(test_pairs)):
            req_idx, fb_idx = test_pairs[i]
            rid = requirements_df.iloc[req_idx, 0]
            fid = feedback_df.iloc[fb_idx, 0]
            if fid in ground_truth_map.get(rid, []):
                test_gt_dict[rid].append(fid)
        max_gt_len = max(len(vals) for vals in test_gt_dict.values()) if test_gt_dict else 0
        for rid in test_gt_dict:
            test_gt_dict[rid] += [''] * (max_gt_len - len(test_gt_dict[rid]))
        df_test_gt = pd.DataFrame(test_gt_dict)
        single_gt_path = os.path.join(results_dir, f"test_ground_truth_single_exp_{exp_id}.xlsx")
        df_test_gt.to_excel(single_gt_path, index=False)

        # Compute FeReRe metrics
        p, r, f2, avg_assign = compute_metrics(df_classified, df_test_gt)

        print(f"[Single Run] => Precision={p:.4f}, Recall={r:.4f}, F2={f2:.4f}, AvgAssigned={avg_assign:.2f}")
        mlflow.log_metric("precision", p)
        mlflow.log_metric("recall", r)
        mlflow.log_metric("f2", f2)
        mlflow.log_metric("avg_assigned", avg_assign)

    return {"precision": p, "recall": r, "f2": f2, "avg_assigned": avg_assign}


if __name__ == "__main__":

    feedback_files = [
        "./data/smartage/SmartAgeSV_Feedback.xlsx",
        "./data/smartage/SmartAgeSF_Feedback.xlsx",
        "./data/komoot/AppReviews.xlsx",
        "./data/ReFeed/feedback.xlsx"
    ]
    requirements_files = [
        "./data/smartage/SV_issues.xlsx",
        "./data/smartage/SF_issues.xlsx",
        "./data/komoot/jira_issues_noprefix.xlsx",
        "./data/ReFeed/requirements.xlsx"
    ]
    ground_truth_files = [
        "./data/smartage/SmartAgeSV_GT_formatted.xlsx",
        "./data/smartage/SmartAgeSF_GT_formatted.xlsx",
        "./data/komoot/Komoot_Ground_Truth_ids_only.xlsx",
        "./data/ReFeed/refeed_gt.xlsx"
    ]

    metrics = train_eval_single_combined(
        feedback_files=feedback_files,
        requirements_files=requirements_files,
        ground_truth_files=ground_truth_files,
        epochs=4,
        batch_size=32,
        max_length=128,
        model_name="bert-base-uncased",
        remove_sw=False,  
        augment=False,    
        do_split=False,   
        random_state=42,
        log_dir="./logs_single",
        results_dir="./single_results",
        exp_name="FeReRe_Single_Combined",
        exp_id=1
    )

    print("[SINGLE RUN EXPERIMENT COMPLETE]")
