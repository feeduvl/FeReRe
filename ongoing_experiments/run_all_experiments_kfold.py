from transformers import logging
logging.set_verbosity_error()

import os
import shutil
import random
import pandas as pd
import numpy as np
import nltk
import mlflow

from pathlib import Path
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from tensorflow.keras import backend as K
import tensorflow as tf

# Import your custom K-fold training function
from finetuneBERT_kfold import train_and_eval_kfold
from calculate_FeReRe_Metrics import run_eval  

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# GLOBAL STOPWORDS
stop_words = set(stopwords.words('english'))

# TEXT PREPROCESSING FUNCTIONS
def remove_stopwords(text: str) -> str:
    words = word_tokenize(text)
    filtered_words = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered_words)

def get_synonym(word):
    synsets = wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != word.lower():
                return lemma_name
    return None

def augment_data_with_synonyms(text: str, replacement_prob=0.15) -> str:
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
    sents = sent_tokenize(text)
    return " || ".join(sents)

def apply_preprocessing(feedback_file, requirements_file, gt_file,
                        remove_sw=False, augment=False, split=False):
    """
    Applies the chosen preprocessing steps to feedback & requirements,
    then saves them into a temporary folder, returning new file paths.
    """
    feedback_df = pd.read_excel(feedback_file, header=None)
    requirements_df = pd.read_excel(requirements_file, header=None)
    gt_df = pd.read_excel(gt_file)

    # Preprocess only text columns: (column index = 1)
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

    tmp_dir = Path("/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/temp_experiments_kfold")
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

def incorporate_previously_assigned_feedback_kfold(
    feedback_file, 
    requirements_file, 
    gt_file, 
    previously_assigned_file
):
    """
    Merges wide-format ground truth with previously assigned feedback IDs.
    The top row has requirement IDs; subsequent rows have feedback IDs.
    We union them for each requirement ID.
    """
    print("[K-FOLD] Reading ground truth (wide format):", gt_file)
    gt_df = pd.read_excel(gt_file, header=None)
    print("[K-FOLD] Reading previously assigned (wide format):", previously_assigned_file)
    prev_df = pd.read_excel(previously_assigned_file, header=None)

    # The top row: requirement IDs
    gt_req_ids = gt_df.iloc[0].dropna().tolist()
    prev_req_ids = prev_df.iloc[0].dropna().tolist()

    # Convert each wide DF to dict { req_id : set(feedback_ids) }
    gt_dict = {}
    for col_idx, req_id in enumerate(gt_req_ids):
        feedback_list = gt_df.iloc[1:, col_idx].dropna().tolist()
        gt_dict[req_id] = set(feedback_list)

    prev_dict = {}
    for col_idx, req_id in enumerate(prev_req_ids):
        feedback_list = prev_df.iloc[1:, col_idx].dropna().tolist()
        prev_dict[req_id] = set(feedback_list)

    # Merge them
    for req_id in prev_dict:
        if req_id in gt_dict:
            gt_dict[req_id] = gt_dict[req_id].union(prev_dict[req_id])
        else:
            gt_dict[req_id] = prev_dict[req_id]

    # Rebuild wide DataFrame
    all_req_ids = list(gt_dict.keys())
    max_fb_count = max(len(fb_set) for fb_set in gt_dict.values()) if gt_dict else 0

    merged_df = pd.DataFrame(index=range(max_fb_count + 1), columns=range(len(all_req_ids)))
    for col_idx, req_id in enumerate(all_req_ids):
        merged_df.iat[0, col_idx] = req_id
    for col_idx, req_id in enumerate(all_req_ids):
        fb_list = list(gt_dict[req_id])
        for row_idx, fb_id in enumerate(fb_list, start=1):
            merged_df.iat[row_idx, col_idx] = fb_id

    expanded_gt = "/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/temp_experiments_kfold/expanded_ground_truth_kfold.xlsx"
    merged_df.to_excel(expanded_gt, header=False, index=False)

    print("[K-FOLD] Merged ground truth with previously assigned. New file:", expanded_gt)
    return feedback_file, requirements_file, expanded_gt


def run_experiment_kfold(
    exp_id,
    remove_sw=False,
    augment=False,
    split=False,
    feedback_files=None,
    requirements_files=None,
    gt_files=None,
    epochs=30,
    n_splits=5,
    batch_size=64,
    model_name='bert-base-uncased',
    subset_size=None
):
    """
    This function applies preprocessing, runs K-fold training/eval,
    and logs only FeReRe metrics (Precision, Recall, F2, Average Assigned) in MLflow.
    """
    print(f"[K-FOLD] Running Experiment {exp_id} with remove_sw={remove_sw}, "
          f"augment={augment}, split={split}, epochs={epochs}, n_splits={n_splits}")

    # 1) Preprocessing
    f_path, r_path, g_path = apply_preprocessing(
        feedback_files[0],
        requirements_files[0],
        gt_files[0],
        remove_sw=remove_sw,
        augment=augment,
        split=split
    )
    
    
    # 2) Define result directories
    results_dir = Path("/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/experiment_results_kfold")
    results_dir.mkdir(exist_ok=True)

    # 3) Start MLflow experiment run
    mlflow.set_experiment("FeReRe_KFold_Experiments")
    with mlflow.start_run(run_name=f"Exp_{exp_id}_kfold",nested=True):
        # Disable automatic logging
        mlflow.tensorflow.autolog(disable=True)
        
        # Log hyperparams
        mlflow.log_param("exp_id", exp_id)
        mlflow.log_param("remove_sw", remove_sw)
        mlflow.log_param("augment", augment)
        mlflow.log_param("split", split)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_name", model_name)
        if subset_size is not None:
            mlflow.log_param("subset_size", subset_size)

        # 4) Run K-fold training/evaluation, returning FeReRe metrics for each fold
        folds_metrics = train_and_eval_kfold(
            feedback=f_path,
            requirements=r_path,
            ground_truth=g_path,
            epochs=epochs,
            n_splits=n_splits,
            batch_size=batch_size,
            model_name=model_name,
            subset_size=subset_size,
            exp_id=exp_id
        )
        
        # 5) Copy each fold's results to the results directory
        #
        # For each fold we have:
        #   /nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/classified_feedback_requirements_{exp_id}_fold_{fold}.xlsx
        #   /nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/test_ground_truth_{exp_id}_fold_{fold}.xlsx
        #
        for fm in folds_metrics:
            fold_id = fm['fold']
            fold_classified_path = (
                f"/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/"
                f"classified_feedback_requirements_{exp_id}_fold_{fold_id}.xlsx"
            )
            fold_gt_path = (
                f"/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/"
                f"test_ground_truth_{exp_id}_fold_{fold_id}.xlsx"
            )

            if os.path.exists(fold_classified_path):
                exp_classified = results_dir / f"classified_feedback_requirements_exp_{exp_id}_fold_{fold_id}.xlsx"
                shutil.copy(fold_classified_path, exp_classified)

            if os.path.exists(fold_gt_path):
                exp_gt = results_dir / f"test_ground_truth_exp_{exp_id}_fold_{fold_id}.xlsx"
                shutil.copy(fold_gt_path, exp_gt)

         # (Optional) Evaluate each fold's final results with run_eval 
            # to log them in a separate MLflow experiment or add more info
            #
            #run_eval(str(exp_classified), str(exp_gt), run_name=f"Exp_{exp_id}_Fold_{fold_id}")
        
        

        
            # Only the four relevant metrics from calculate_FeReRe_Metrics
            # mlflow.log_metric("precision_fold", fm['precision'], step=fold_index)
            # mlflow.log_metric("recall_fold", fm['recall'], step=fold_index)
            # mlflow.log_metric("f2_fold", fm['f2'], step=fold_index)
            # mlflow.log_metric("avg_assign_fold", fm['avg_assign'], step=fold_index)
            mlflow.log_metric(f"precision_fold_{fold_id}", fm['precision'])
            mlflow.log_metric(f"recall_fold_{fold_id}", fm['recall'])
            mlflow.log_metric(f"f2_fold_{fold_id}", fm['f2'])
            mlflow.log_metric(f"avg_assign_fold_{fold_id}", fm['avg_assign'])


        # Optionally, log average across folds (same four metrics)
        if folds_metrics:
            overall_precision = np.mean([fm['precision'] for fm in folds_metrics])
            overall_recall = np.mean([fm['recall'] for fm in folds_metrics])
            overall_f2 = np.mean([fm['f2'] for fm in folds_metrics])
            overall_avg_feedback = np.mean([fm['avg_assign'] for fm in folds_metrics])
            
            # Print only the overall averages.
            # print("\nOverall Performance:")
            # print(f"Precision: {overall_precision:.2f}")
            # print(f"Recall: {overall_recall:.2f}")
            # print(f"F2 Score: {overall_f2:.2f}")
            # print(f"Average Feedback per Requirement: {overall_avg_feedback:.2f}")
            
            # Log overall metrics to MLflow.
            mlflow.log_metric("precision", overall_precision)
            mlflow.log_metric("recall", overall_recall)
            mlflow.log_metric("f2_score", overall_f2)
            mlflow.log_metric("avg_assign", overall_avg_feedback)

    # 5) Clean up TensorFlow session
    K.clear_session()
    tf.compat.v1.reset_default_graph()

    print(f"[K-FOLD] Experiment {exp_id} completed.\n")


if __name__ == "__main__":
    # Example input files
    feedback_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/AppReviews.xlsx"]
    requirements_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/jira_issues_noprefix.xlsx"]
    gt_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/Komoot_Ground_Truth_ids_only.xlsx"]

    # # 1) Stopword Removal
    # run_experiment_kfold(
    #     exp_id=1,
    #     remove_sw=True,
    #     augment=False,
    #     split=False,
    #     feedback_files=feedback_files,
    #     requirements_files=requirements_files,
    #     gt_files=gt_files,
       
       
    #     model_name='bert-base-uncased'
    # )

    # 2) Data Augmentation
    run_experiment_kfold(
        exp_id=2,
        remove_sw=False,
        augment=True,
        split=False,
        feedback_files=feedback_files,
        requirements_files=requirements_files,
        gt_files=gt_files,
        
        model_name='bert-base-uncased'
    )

    # 3) Sentence Splitting
    run_experiment_kfold(
        exp_id=3,
        remove_sw=False,
        augment=False,
        split=True,
        feedback_files=feedback_files,
        requirements_files=requirements_files,
        gt_files=gt_files,
       
        model_name='bert-base-uncased'
    )

    # # 4) Stopword Removal + Data Augmentation
    # run_experiment_kfold(
    #     exp_id=4,
    #     remove_sw=True,
    #     augment=True,
    #     split=False,
    #     feedback_files=feedback_files,
    #     requirements_files=requirements_files,
    #     gt_files=gt_files,
       
    #     model_name='bert-base-uncased'
    # )

    # # 5) Stopword Removal + Sentence Splitting
    # run_experiment_kfold(
    #     exp_id=5,
    #     remove_sw=True,
    #     augment=False,
    #     split=True,
    #     feedback_files=feedback_files,
    #     requirements_files=requirements_files,
    #     gt_files=gt_files,
        
    #     model_name='bert-base-uncased'
    # )

    # # 6) Data Augmentation + Sentence Splitting
    # run_experiment_kfold(
    #     exp_id=6,
    #     remove_sw=False,
    #     augment=True,
    #     split=True,
    #     feedback_files=feedback_files,
    #     requirements_files=requirements_files,
    #     gt_files=gt_files,
        
    #     model_name='bert-base-uncased'
    # )

    # # 7) All three
    # run_experiment_kfold(
    #     exp_id=7,
    #     remove_sw=True,
    #     augment=True,
    #     split=True,
    #     feedback_files=feedback_files,
    #     requirements_files=requirements_files,
    #     gt_files=gt_files,
       
    #     model_name='bert-base-uncased'
    # )

    # # #
    # # # Suppose you found the best preprocessing is Exp_2
    # # #
    # # best_preprocessing = {"remove_sw": False, "augment": True, "split": False}

    # # #
    # # # 8–13: Different models with best preprocessing
    # # #
    # # # 8: RoBERTa without Preprocessing
    # # run_experiment_kfold(
    # #     exp_id=8,
    # #     remove_sw=False,
    # #     augment=False,
    # #     split=False,
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="roberta-base"
    # # )

    # # # 9: RoBERTa with best preprocessing
    # # run_experiment_kfold(
    # #     exp_id=9,
    # #     remove_sw=best_preprocessing["remove_sw"],
    # #     augment=best_preprocessing["augment"],
    # #     split=best_preprocessing["split"],
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="roberta-base"
    # # )

    # # # 10: BERT-Large without Preprocessing
    # # run_experiment_kfold(
    # #     exp_id=10,
    # #     remove_sw=False,
    # #     augment=False,
    # #     split=False,
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="bert-large-uncased"
    # # )

    # # # 11: BERT-Large with best preprocessing
    # # run_experiment_kfold(
    # #     exp_id=11,
    # #     remove_sw=best_preprocessing["remove_sw"],
    # #     augment=best_preprocessing["augment"],
    # #     split=best_preprocessing["split"],
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="bert-large-uncased"
    # # )

    # # # 12: DistilBERT without Preprocessing
    # # run_experiment_kfold(
    # #     exp_id=12,
    # #     remove_sw=False,
    # #     augment=False,
    # #     split=False,
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="distilbert-base-uncased"
    # # )

    # # # 13: DistilBERT with best preprocessing
    # # run_experiment_kfold(
    # #     exp_id=13,
    # #     remove_sw=best_preprocessing["remove_sw"],
    # #     augment=best_preprocessing["augment"],
    # #     split=best_preprocessing["split"],
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="distilbert-base-uncased"
    # # )

    # # #
    # # # 15: BERT-Base baseline (no preprocessing)
    # # #
    # # run_experiment_kfold(
    # #     exp_id=15,
    # #     remove_sw=False,
    # #     augment=False,
    # #     split=False,
    # #     feedback_files=feedback_files,
    # #     requirements_files=requirements_files,
    # #     gt_files=gt_files,
    # #     epochs=2,
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name='bert-base-uncased'
    # # )

    # # #
    # # # 14: Incorporate Previously Assigned Feedback
    # # #
    # # prev_assigned = "/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/experiment_results_kfold/classified_feedback_requirements_exp_2.xlsx"

    # # # 1) Merge old feedback with GT (wide format)
    # # f_path, r_path, g_path = incorporate_previously_assigned_feedback_kfold(
    # #     feedback_files[0],
    # #     requirements_files[0],
    # #     gt_files[0],
    # #     prev_assigned
    # # )

    # # # 2) Apply best preprocessing
    # # f_path_pre, r_path_pre, g_path_pre = apply_preprocessing(
    # #     f_path,
    # #     r_path,
    # #     g_path,
    # #     remove_sw=best_preprocessing["remove_sw"],
    # #     augment=best_preprocessing["augment"],
    # #     split=best_preprocessing["split"]
    # # )

    # # # 3) Train/Eval with K-fold
    # # print("Running Experiment 14 with previously assigned feedback (Exp_2).")
    # # folds_metrics = train_and_eval_kfold(
    # #     feedback=f_path_pre,
    # #     requirements=r_path_pre,
    # #     ground_truth=g_path_pre,
    # #     epochs=2,  # or 30, up to you
    # #     n_splits=2,
    # #     batch_size=64,
    # #     model_name="bert-base-uncased",
    # #     exp_id=14
    # # )

    # # # 4) Evaluate final results or copy files 
    
    # # classified_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/classified_feedback_requirements.xlsx'
    # # gt_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/test_ground_truth.xlsx'
    # # run_eval(classified_file, gt_file, run_name="Experiment 14 K-Fold")

    # # print("[K-FOLD] All experiments finished.")
