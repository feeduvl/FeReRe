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

    tmp_dir = Path("temp_experiments")
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
                   feedback_files=None, requirements_files=None, gt_files=None, epochs=1):
    print(f"Running Experiment {exp_id}: model={model_name}, remove_sw={remove_sw}, augment={augment}, split={split}")
    f_path, r_path, g_path = apply_preprocessing(feedback_files[0], requirements_files[0], gt_files[0],
                                                 remove_sw=remove_sw, augment=augment, split=split)

    # Increase subset_size or remove it entirely for full data
    train_and_eval_bert(f_path, r_path, g_path, epochs=epochs, model_name=model_name)

    results_dir = Path("experiment_results")
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

def incorporate_previously_assigned_feedback(feedback_file, requirements_file, gt_file, previously_assigned_file):
    gt_df = pd.read_excel(gt_file)
    prev_df = pd.read_excel(previously_assigned_file)
    # Example logic: just return original for now.
    expanded_gt = "temp_experiments/expanded_ground_truth.xlsx"
    gt_df.to_excel(expanded_gt, index=False)
    return feedback_file, requirements_file, expanded_gt

if __name__ == "__main__":
    feedback_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/AppReviews.xlsx"]
    requirements_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/jira_issues_noprefix.xlsx"]
    gt_files = ["/nfs/home/vthakur_paech/FeReRe/data/komoot/Komoot_Ground_Truth_ids_only.xlsx"]

    # Experiments 1-7: BERT-Base with various preprocessing
    # 1: Stopword Removal
    run_experiment(1, model_name="bert-base-uncased", remove_sw=True, augment=False, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 2: Data Augmentation
    run_experiment(2, model_name="bert-base-uncased", remove_sw=False, augment=True, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 3: Sentence Splitting
    run_experiment(3, model_name="bert-base-uncased", remove_sw=False, augment=False, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 4: Stopword Removal + Data Augmentation
    run_experiment(4, model_name="bert-base-uncased", remove_sw=True, augment=True, split=False,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 5: Stopword Removal + Sentence Splitting
    run_experiment(5, model_name="bert-base-uncased", remove_sw=True, augment=False, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 6: Data Augmentation + Sentence Splitting
    run_experiment(6, model_name="bert-base-uncased", remove_sw=False, augment=True, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)
    # 7: All three
    run_experiment(7, model_name="bert-base-uncased", remove_sw=True, augment=True, split=True,
                   feedback_files=feedback_files, requirements_files=requirements_files, gt_files=gt_files)

    # After reviewing experiments 1-7, determine the best preprocessing:
    # Let's assume best_preprocessing is found:
    best_preprocessing = {"remove_sw": False, "augment": False, "split": True}

    # Experiments 8-13: Different models with best preprocessing
    # 8: RoBERTa without Preprocessing
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

    # Experiment 14: Incorporate Previously Assigned Feedback
    prev_assigned = "experiment_results/classified_feedback_requirements_exp_11.xlsx"
    f_path, r_path, g_path = incorporate_previously_assigned_feedback(feedback_files[0], requirements_files[0],
                                                                     gt_files[0], prev_assigned)
    f_path_pre, r_path_pre, g_path_pre = apply_preprocessing(f_path, r_path, g_path,
                                                             remove_sw=best_preprocessing["remove_sw"],
                                                             augment=best_preprocessing["augment"],
                                                             split=best_preprocessing["split"])
    print("Running Experiment 14: Incorporating Previously Assigned Feedback")
    # Train with best-performing model (let's assume bert-large-uncased)
    train_and_eval_bert(f_path_pre, r_path_pre, g_path_pre, epochs=1, model_name="bert-large-uncased")

    classified_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/classified_feedback_requirements.xlsx'
    gt_file = '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/test_ground_truth.xlsx'
    exp_classified = Path("experiment_results") / "classified_feedback_requirements_exp_14.xlsx"
    exp_gt = Path("experiment_results") / "test_ground_truth_exp_14.xlsx"
    if os.path.exists(classified_file):
        shutil.copy(classified_file, exp_classified)
    if os.path.exists(gt_file):
        shutil.copy(gt_file, exp_gt)
    run_eval(str(exp_classified), str(exp_gt), run_name="Experiment 14")
    print("Experiment 14 completed.")

    print("All experiments finished.")
