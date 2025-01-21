"""
run_all_exp_combined_data.py

This script runs a series of K-fold experiments (15 configurations) on the combined dataset
using the function train_eval_kfold_combined from kfold_combined.py. It applies different preprocessing
and model settings and logs FeReRe metrics via MLflow.

Additionally, it includes the helper function incorporate_previously_assigned_feedback_kfold to
merge previously assigned feedback into the ground truth (demonstrated in experiment 14).

RESULT FILES SAVED:
  For each experiment (identified by its exp_id) and for each fold:
    - Classified Feedback (wide-format):
         classified_feedback_requirements_exp_<exp_id>_fold_<fold>.xlsx
    - Test Ground Truth (wide-format):
         test_ground_truth_exp_<exp_id>_fold_<fold>.xlsx
  These files are saved in the results directory (e.g., "./kfold_results").
  
  MLflow logs the parameters and metrics for each experiment.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import logging
logging.set_verbosity_error()

import random
import mlflow
from kfold_fintuneBERT_combined import train_eval_kfold_combined
import pandas as pd
from pathlib import Path

# -------------------------------
# incorporate_previously_assigned_feedback_kfold Function
# -------------------------------
def incorporate_previously_assigned_feedback_kfold(
    feedback_file, 
    requirements_file, 
    gt_file, 
    previously_assigned_file
):
    """
    Merges wide-format ground truth with previously assigned feedback IDs.
    The top row of each file contains requirement IDs; subsequent rows contain feedback IDs.
    
    For each requirement, one randomly chosen previously assigned feedback (if available)
    is incorporated into the ground truth.
    
    The function saves the merged (expanded) ground truth to a new Excel file and returns:
       (feedback_file, requirements_file, path_to_merged_gt_file)
    """
    print("[K-FOLD] Reading ground truth (wide format):", gt_file)
    gt_df = pd.read_excel(gt_file, header=None)
    print("[K-FOLD] Reading previously assigned feedback (wide format):", previously_assigned_file)
    prev_df = pd.read_excel(previously_assigned_file, header=None)

    # The top row contains requirement IDs.
    gt_req_ids = gt_df.iloc[0].dropna().tolist()
    prev_req_ids = prev_df.iloc[0].dropna().tolist()

    # Convert each wide DataFrame to a dictionary: { req_id : set(feedback_ids) }
    gt_dict = {}
    for col_idx, req_id in enumerate(gt_req_ids):
        feedback_list = gt_df.iloc[1:, col_idx].dropna().tolist()
        gt_dict[req_id] = set(feedback_list)

    prev_dict = {}
    for col_idx, req_id in enumerate(prev_req_ids):
        feedback_list = prev_df.iloc[1:, col_idx].dropna().tolist()
        prev_dict[req_id] = set(feedback_list)

    # For each requirement in prev_dict, if there is at least one feedback, add one randomly chosen feedback into gt_dict.
    for req_id in prev_dict:
        if len(prev_dict[req_id]) > 0:
            one_random_fb = random.choice(list(prev_dict[req_id]))
            if req_id not in gt_dict:
                gt_dict[req_id] = set()
            gt_dict[req_id].add(one_random_fb)

    # Rebuild a wide DataFrame from the updated gt_dict
    all_req_ids = list(gt_dict.keys())
    max_fb_count = max(len(fb_set) for fb_set in gt_dict.values()) if gt_dict else 0
    merged_df = pd.DataFrame(index=range(max_fb_count + 1), columns=range(len(all_req_ids)))
    for col_idx, req_id in enumerate(all_req_ids):
        merged_df.iat[0, col_idx] = req_id
    for col_idx, req_id in enumerate(all_req_ids):
        fb_list = list(gt_dict[req_id])
        for row_idx, fb_id in enumerate(fb_list, start=1):
            merged_df.iat[row_idx, col_idx] = fb_id

    # Save the merged ground truth to a new Excel file in a temporary folder.
    tmp_dir = Path("./ongoing_experiments/temp_experiments_kfold")
    tmp_dir.mkdir(exist_ok=True)
    expanded_gt = tmp_dir / "expanded_ground_truth_kfold.xlsx"
    merged_df.to_excel(expanded_gt, header=False, index=False)

    print("[K-FOLD] Merged ground truth with previously assigned feedback saved to:", expanded_gt)
    return feedback_file, requirements_file, str(expanded_gt)



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


experiments = [
    {"exp_id": 1,  "remove_sw": True,  "augment": False, "do_split": False,"epochs":30, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
    {"exp_id": 2,  "remove_sw": False, "augment": True,  "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
    {"exp_id": 3,  "remove_sw": False, "augment": False, "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
    {"exp_id": 4,  "remove_sw": True,  "augment": True,  "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
    {"exp_id": 5,  "remove_sw": True,  "augment": False, "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
    {"exp_id": 6,  "remove_sw": False, "augment": True,  "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},
    {"exp_id": 7,  "remove_sw": True,  "augment": True,  "do_split": True,  "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"}
   # Use different models with best preprocessing (best: remove_sw=False, augment=True, do_split=False)
   
    {"exp_id": 8,  "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "roberta-base"},
    {"exp_id": 9,  "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "roberta-base"},
    {"exp_id": 10, "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-large-uncased"},
    {"exp_id": 11, "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-large-uncased"},
    {"exp_id": 12, "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "distilbert-base-uncased"},
    {"exp_id": 13, "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "distilbert-base-uncased"},
    #Exp_15 is the baseline exp
    {"exp_id": 15, "remove_sw": False, "augment": False, "do_split": False, "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"},

    # # Experiment 14: Incorporate previously assigned feedback (using best_preprocessing)
    {"exp_id": 14, "remove_sw": best_preprocessing["remove_sw"], "augment": best_preprocessing["augment"], "do_split": best_preprocessing["do_split"], "epochs": 20, "n_splits": 5, "batch_size": 32, "model_name": "bert-base-uncased"}
    
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
        
        if exp["exp_id"] == 14:
                
            # For Experiment 14, choose a random fold file from Exp 2's outputs.
            random_fold = random.randint(1, 5)
            prev_assigned = f"./kfold_results/classified_feedback_requirements_exp_2_fold_{random_fold}.xlsx"
            
            f_path, r_path, g_path = incorporate_previously_assigned_feedback_kfold(
                feedback_files[0],
                requirements_files[0],
                ground_truth_files[0],
                prev_assigned
            )
            # For experiment 14, use the merged ground truth file and best_preprocessing parameters.
            _ = train_eval_kfold_combined(
                feedback_files=[f_path],
                requirements_files=[r_path],
                ground_truth_files=[g_path],
                n_splits=exp["n_splits"],
                epochs=exp["epochs"],
                batch_size=exp["batch_size"],
                max_length=256,
                model_name=exp["model_name"],
                remove_sw=best_preprocessing["remove_sw"],
                augment=best_preprocessing["augment"],
                do_split=best_preprocessing["do_split"],
                random_state=42,
                log_dir="./logs_kfold",
                results_dir="./kfold_results",
                exp_name="FeReRe_KFold_Combined",
                exp_id=exp["exp_id"]
            )
        else:
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
                log_dir="./logs_kfold",
                results_dir="./kfold_results",
                exp_name="FeReRe_KFold_Combined",
                exp_id=exp["exp_id"]
            )

    print("\n[ALL EXPERIMENTS COMPLETED]\n")

if __name__ == "__main__":
    run_all_experiments()