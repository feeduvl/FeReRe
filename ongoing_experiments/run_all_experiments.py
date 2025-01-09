from transformers import logging
logging.set_verbosity_error()
import os
import shutil
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from finetuneBERT import train_and_eval as train_and_eval_bert
from calculate_FeReRe_Metrics import run_eval
import random
from tensorflow.keras import backend as K
import tensorflow as tf
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Global stopwords set
stop_words = set(stopwords.words('english'))

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
    feedback_df = pd.read_excel(feedback_file, header=None)
    requirements_df = pd.read_excel(requirements_file, header=None)
    gt_df = pd.read_excel(gt_file)

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

    tmp_dir = Path("/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/temp_experiments")
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

def run_experiment(exp_id, model_name="bert-base-uncased",
                   remove_sw=False, augment=False, split=False,
                   feedback_files=None, requirements_files=None, gt_files=None, epochs=30):
    
    try:
        print(f"Running Experiment {exp_id}: model={model_name}, remove_sw={remove_sw}, augment={augment}, split={split}")
        f_path, r_path, g_path = apply_preprocessing(feedback_files[0], requirements_files[0], gt_files[0],
                                                    remove_sw=remove_sw, augment=augment, split=split)

        # Increase subset_size or remove it entirely for full data
        train_and_eval_bert(f_path, r_path, g_path, epochs=epochs, model_name=model_name,  exp_id=exp_id)

        results_dir = Path("/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/experiment_results")
        results_dir.mkdir(exist_ok=True)

        classified_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/classified_feedback_requirements.xlsx'
        gt_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/test_ground_truth.xlsx'

        exp_classified = results_dir / f"classified_feedback_requirements_exp_{exp_id}.xlsx"
        exp_gt = results_dir / f"test_ground_truth_exp_{exp_id}.xlsx"

        if os.path.exists(classified_file):
            shutil.copy(classified_file, exp_classified)
        if os.path.exists(gt_file):
            shutil.copy(gt_file, exp_gt)

        run_eval(str(exp_classified), str(exp_gt), run_name=f"Experiment {exp_id}")
        print(f"Experiment {exp_id} completed.")
        
    
    finally:
        
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        
        


# def incorporate_previously_assigned_feedback(feedback_file, requirements_file, gt_file, previously_assigned_file):
#     gt_df = pd.read_excel(gt_file)
#     prev_df = pd.read_excel(previously_assigned_file)
#     # Example logic: just return original for now.
#     expanded_gt = "/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/temp_experiments/expanded_ground_truth.xlsx"
#     gt_df.to_excel(expanded_gt, index=False)
#     return feedback_file, requirements_file, expanded_gt


def incorporate_previously_assigned_feedback(
    feedback_file, 
    requirements_file, 
    gt_file, 
    previously_assigned_file
):
    """
    Wide-format merge for Komoot data:
     - The first row of each file has requirement IDs as columns (e.g., "KOMOOT-8", "KOMOOT-66", etc.).
     - Below each column are the feedback IDs. 
    We union the feedback IDs from ground truth and previously assigned file.
    
    Returns
    -------
    (feedback_file, requirements_file, expanded_gt_path)
    """
    print("Reading ground truth (wide format):", gt_file)
    gt_df = pd.read_excel(gt_file, header=None)
    print("Reading previously assigned feedback (wide format):", previously_assigned_file)
    prev_df = pd.read_excel(previously_assigned_file, header=None)

    # 1) The top row: requirement IDs
    gt_req_ids = gt_df.iloc[0].dropna().tolist()
    prev_req_ids = prev_df.iloc[0].dropna().tolist()

    # 2) Convert each wide DF into a dict { requirement_id : set_of_feedback_ids }
    gt_dict = {}
    for col_idx, req_id in enumerate(gt_req_ids):
        feedback_list = gt_df.iloc[1:, col_idx].dropna().tolist()
        gt_dict[req_id] = set(feedback_list)

    prev_dict = {}
    for col_idx, req_id in enumerate(prev_req_ids):
        feedback_list = prev_df.iloc[1:, col_idx].dropna().tolist()
        prev_dict[req_id] = set(feedback_list)

    # 3) Merge the previously assigned feedback into the ground truth
    for req_id in prev_dict:
        if req_id in gt_dict:
            gt_dict[req_id] = gt_dict[req_id].union(prev_dict[req_id])
        else:
            # If the requirement isn't in GT, we add it as a brand-new column
            gt_dict[req_id] = prev_dict[req_id]

    # 4) Rebuild a wide DataFrame with all requirement columns
    all_req_ids = list(gt_dict.keys())
    max_feedback_count = max(len(fb_set) for fb_set in gt_dict.values()) if gt_dict else 0

    merged_df = pd.DataFrame(index=range(max_feedback_count + 1), columns=range(len(all_req_ids)))

    # Top row = requirement IDs
    for col_idx, req_id in enumerate(all_req_ids):
        merged_df.iat[0, col_idx] = req_id

    # Fill in feedback IDs in rows below
    for col_idx, req_id in enumerate(all_req_ids):
        fb_list = list(gt_dict[req_id])
        for row_idx, fb_id in enumerate(fb_list, start=1):
            merged_df.iat[row_idx, col_idx] = fb_id

    # 5) Save the new expanded ground truth
    expanded_gt = "/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/temp_experiments/expanded_ground_truth.xlsx"
    merged_df.to_excel(expanded_gt, header=False, index=False)

    print(f"[INFO] Successfully merged wide-format GT with previously assigned. New file: {expanded_gt}")
    return feedback_file, requirements_file, expanded_gt


if __name__ == "__main__":
    
    feedback_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/AppReviews.xlsx"]
    requirements_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/jira_issues_noprefix.xlsx"]
    gt_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/Komoot_Ground_Truth_ids_only.xlsx"]

    # Experiments 1-7: BERT-Base with various preprocessing
    
    # # 1: Stopword Removal
    run_experiment(1, model_name="bert-base-uncased", remove_sw=True, augment=False, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # # 2: Data Augmentation
    run_experiment(2, model_name="bert-base-uncased", remove_sw=False, augment=True, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 3: Sentence Splitting
    run_experiment(3, model_name="bert-base-uncased", remove_sw=False, augment=False, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # # 4: Stopword Removal + Data Augmentation
    run_experiment(4, model_name="bert-base-uncased", remove_sw=True, augment=True, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # # # 5: Stopword Removal + Sentence Splitting
    run_experiment(5, model_name="bert-base-uncased", remove_sw=True, augment=False, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # # # 6: Data Augmentation + Sentence Splitting
    run_experiment(6, model_name="bert-base-uncased", remove_sw=False, augment=True, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # # 7: All three
    run_experiment(7, model_name="bert-base-uncased", remove_sw=True, augment=True, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)

    # # After reviewing experiments 1-7, determine the best preprocessing:
    # # Accoriding to results obtained best_preprocessing is exp_2:
    best_preprocessing = {"remove_sw": False, "augment": True, "split": False}

    # # Experiments 8-13: Different models with best preprocessing
    # # 8: RoBERTa without Preprocessing
    run_experiment(8, model_name="roberta-base", remove_sw=False, augment=False, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 9: RoBERTa with Best Preprocessing
    run_experiment(9, model_name="roberta-base", remove_sw=best_preprocessing["remove_sw"],
                   augment=best_preprocessing["augment"], split=best_preprocessing["split"],
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 10: BERT-Large without Preprocessing
    run_experiment(10, model_name="bert-large-uncased", remove_sw=False, augment=False, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 11: BERT-Large with Best Preprocessing
    run_experiment(11, model_name="bert-large-uncased", remove_sw=best_preprocessing["remove_sw"],
                   augment=best_preprocessing["augment"], split=best_preprocessing["split"],
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 12: DistilBERT without Preprocessing
    run_experiment(12, model_name="distilbert-base-uncased", remove_sw=False, augment=False, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 13: DistilBERT with Best Preprocessing
    run_experiment(13, model_name="distilbert-base-uncased", remove_sw=best_preprocessing["remove_sw"],
                   augment=best_preprocessing["augment"], split=best_preprocessing["split"],
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    
    #15: BERT-Base (baaseline_exp without preprocessing)
    run_experiment(15, model_name="bert-base-uncased", remove_sw=False, augment=False, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
  

    # Experiment 14: Incorporate Previously Assigned Feedback
    prev_assigned = "/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/experiment_results/classified_feedback_requirements_exp_2.xlsx"
    

# 1. Incorporate previously assigned feedback into GT
    f_path, r_path, g_path = incorporate_previously_assigned_feedback(
        feedback_files[0],
        requirements_files[0],
        gt_files[0],
        prev_assigned
    )

    # 2.  apply your best preprocessing
     f_path_pre, r_path_pre, g_path_pre = apply_preprocessing(
        f_path, r_path, g_path,
        remove_sw=best_preprocessing["remove_sw"],
        augment=best_preprocessing["augment"],
        split=best_preprocessing["split"]
    )

    # 3. Train and evaluate as usual
    print("Running Experiment 14 with previously assigned feedback from Exp_2.")
    train_and_eval_bert(
        f_path_pre, r_path_pre, g_path_pre,
        epochs=30, model_name="bert-base-uncased", exp_id=14
    )

    # 4. Copy & evaluate final results
    classified_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/classified_feedback_requirements.xlsx'
    gt_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/test_ground_truth.xlsx'
    exp_classified = Path("/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/experiment_results") / "classified_feedback_requirements_exp_14.xlsx"
    exp_gt = Path("/nfs/home/vthakur_paech/FeReRe/ongoing_experiments/experiment_results") / "test_ground_truth_exp_14.xlsx"

    if os.path.exists(classified_file):
        shutil.copy(classified_file, exp_classified)
    if os.path.exists(gt_file):
        shutil.copy(gt_file, exp_gt)

    run_eval(str(exp_classified), str(exp_gt), run_name="Experiment 14")
    print("Experiment 14 completed.")
    

    
    print("All experiments finished.")
