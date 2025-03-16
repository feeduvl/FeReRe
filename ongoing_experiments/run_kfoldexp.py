"""
run_kfoldexp.py

This script runs a series of K-fold experiments (15 configurations) on the combined dataset
using the function train_eval_kfold_combined from kfold_combined.py. It applies different preprocessing
and model settings and logs FeReRe metrics via MLflow.

RESULT FILES SAVED:
  For each experiment (identified by its exp_id) and for each fold:
    - Classified Feedback (wide-format):
         classified_feedback_requirements_exp_<exp_id>_fold_<fold>.xlsx
    - Test Ground Truth (wide-format):
         test_ground_truth_exp_<exp_id>_fold_<fold>.xlsx
  These files are saved in the results directory (e.g., "./new_kfold_results").

  MLflow logs the parameters and metrics for each experiment.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import logging
logging.set_verbosity_error()

import random
import mlflow
from kfold_combined import train_eval_kfold_combined
import pandas as pd
from pathlib import Path

def incorporate_previously_assigned_feedback_kfold(
    feedback_files, 
    requirements_files, 
    ground_truth_files
):
    """
    1) Merge all ground-truth files (wide format).
    2) For each requirement, pick 1 random feedback from ground truth => "already assigned".
    3) Remove that feedback from the ground truth so it's not re-classified.
    4) Append that feedback's text to the requirement text => new requirements file.
    5) Return the paths to the updated requirement file & updated ground truth file.
    """
    # --- A) Combine ground-truth wide data ---
    gt_df_list = []
    for gfile in ground_truth_files:
        gdf = pd.read_excel(gfile, header=None)
        gt_df_list.append(gdf)
    merged_gt = pd.concat(gt_df_list, axis=1, ignore_index=True)

    # Convert to dictionary {req_id : set of fb_ids}
    gt_dict = {}
    for col_idx in range(merged_gt.shape[1]):
        req_id = merged_gt.iat[0, col_idx]
        if pd.notna(req_id):
            fb_set = set( merged_gt.iloc[1:, col_idx].dropna().tolist() )
            gt_dict[req_id] = fb_set

    # --- B) Randomly pick 'already assigned' feedback for each requirement ---
    already_assigned = {}  # {req_id : fb_id}
    for rid in gt_dict:
        fb_set = gt_dict[rid]
        if fb_set:
            chosen_fb = random.choice(list(fb_set))
            already_assigned[rid] = chosen_fb
            gt_dict[rid].remove(chosen_fb)  # remove from classification
        else:
            already_assigned[rid] = None

    # Rebuild ground-truth in wide format (with the chosen FB removed)
    all_req_ids = list(gt_dict.keys())
    max_count = max(len(gt_dict[r]) for r in gt_dict) if gt_dict else 0
    new_gt = pd.DataFrame(index=range(max_count + 1), columns=range(len(all_req_ids)))
    for col_idx, rid in enumerate(all_req_ids):
        new_gt.iat[0, col_idx] = rid
        fb_list = list(gt_dict[rid])
        for row_idx, fb_id in enumerate(fb_list, start=1):
            new_gt.iat[row_idx, col_idx] = fb_id

    # Save new ground truth
    tmp_dir = Path("./ongoing_experiments/temp_experiments_new_kfold")
    tmp_dir.mkdir(exist_ok=True)
    new_gt_path = tmp_dir / "ground_truth_exp14.xlsx"
    new_gt.to_excel(new_gt_path, header=False, index=False)

    # --- C) Append assigned FB's text to the requirements' text ---
    #  1) Merge all requirements files
    req_df_list = []
    for rfile in requirements_files:
        rdf = pd.read_excel(rfile, header=None)
        req_df_list.append(rdf)
    merged_req = pd.concat(req_df_list, axis=0, ignore_index=True)
    #  2) Merge all feedback files (so we can look up text by ID)
    fb_df_list = []
    for ffile in feedback_files:
        fdf = pd.read_excel(ffile, header=None)
        fb_df_list.append(fdf)
    merged_fb = pd.concat(fb_df_list, axis=0, ignore_index=True)
    
    # Build a dict {fb_id => fb_text}
    fb_dict = {}
    for i in range(merged_fb.shape[0]):
        fb_id = merged_fb.iat[i, 0]
        fb_text = str( merged_fb.iat[i, 1] )
        fb_dict[fb_id] = fb_text

    #  3) For each requirement row, if we have already assigned FB, then append it
    updated_req_rows = []
    for i in range(merged_req.shape[0]):
        rid = merged_req.iat[i, 0]
        req_text = str( merged_req.iat[i, 1] )
        if rid in already_assigned and already_assigned[rid] is not None:
            assigned_fb_id = already_assigned[rid]
            assigned_fb_text = fb_dict.get(assigned_fb_id, "")
            # Append
            new_text = req_text + " [PAST_FEEDBACK] " + assigned_fb_text
            updated_req_rows.append([rid, new_text])
        else:
            updated_req_rows.append([rid, req_text])

    updated_req_df = pd.DataFrame(updated_req_rows)
    new_req_path = tmp_dir / "requirements_exp14.xlsx"
    updated_req_df.to_excel(new_req_path, header=False, index=False)

    return str(new_req_path), str(new_gt_path)

# -------------------------------
# Experiment Configurations
# -------------------------------
# Define data file paths (update these as needed)
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

best_preprocessing = {"remove_sw": False, "augment": True, "do_split": False}

# List of experiment configurations (15 experiments):
# For experiments 1-7, 8-13, and 15, we use different settings.
# Experiment 14 uses incorporate_previously_assigned_feedback_kfold.
experiments = [
     {"exp_id": 1,  "remove_sw": True,  "augment": False, "do_split": False,"epochs":20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 2,  "remove_sw": False, "augment": True,  "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 3,  "remove_sw": False, "augment": False, "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 4,  "remove_sw": True,  "augment": True,  "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 5,  "remove_sw": True,  "augment": False, "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 6,  "remove_sw": False, "augment": True,  "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 7,  "remove_sw": True,  "augment": True,  "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 8,  "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "roberta-base"},
     {"exp_id": 9,  "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "roberta-base"},
     {"exp_id": 10, "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-large-uncased"},
     {"exp_id": 11, "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-large-uncased"},
     {"exp_id": 12, "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "distilbert-base-uncased"},
     {"exp_id": 13, "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "distilbert-base-uncased"},
     {"exp_id": 15, "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
     {"exp_id": 14, "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-large-uncased"}
]

# Optionally, set a common MLflow experiment name for all runs
mlflow.set_experiment("FeReRe_KFold_Combined_Experiments")

def run_all_experiments():
    for exp in experiments:
        print("\n==========================================")
        print(f"Running K-fold Experiment {exp['exp_id']} with parameters:")
        print(f"  remove_sw={exp['remove_sw']} | augment={exp['augment']} | do_split={exp['do_split']}")
        print(f"  epochs={exp['epochs']} | n_splits={exp['n_splits']} | batch_size={exp['batch_size']} | model_name={exp['model_name']}")
        print("==========================================\n")
        
        # Normal experiments (1..13, 15)
        if exp["exp_id"] != 14:
            _ = train_eval_kfold_combined(
                feedback_files=feedback_files,
                requirements_files=requirements_files,
                ground_truth_files=ground_truth_files,
                n_splits=exp["n_splits"],
                epochs=exp["epochs"],
                batch_size=exp["batch_size"],
                max_length=256,
                model_name=exp["model_name"],
                remove_sw=exp["remove_sw"],
                augment=exp["augment"],
                do_split=exp["do_split"],
                random_state=42,
                # Removed log_dir parameter (TensorBoard is no longer used)
                results_dir="./new_kfold_results",
                exp_name="FeReRe_KFold_Combined",
                exp_id=exp["exp_id"]
            )
        
        elif exp["exp_id"] == 14:
            # Instead of reading from classifier, do the new approach:
            new_req, new_gt = incorporate_previously_assigned_feedback_kfold(
                feedback_files,
                requirements_files,
                ground_truth_files
            )
            
            metrics = train_eval_kfold_combined(
                feedback_files=feedback_files,
                requirements_files=[new_req],
                ground_truth_files=[new_gt],
                n_splits=exp["n_splits"],
                epochs=exp["epochs"],
                batch_size=exp["batch_size"],
                max_length=256,
                model_name=exp["model_name"],
                remove_sw=exp["remove_sw"],
                augment=exp["augment"],
                do_split=exp["do_split"],
                random_state=42,
                # Removed log_dir parameter (TensorBoard is no longer used)
                results_dir="./new_kfold_results",
                exp_name="FeReRe_KFold_Combined",
                exp_id=exp["exp_id"]
            )

if __name__ == "__main__":
    run_all_experiments()
