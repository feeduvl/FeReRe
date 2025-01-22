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
    feedback_files, 
    requirements_files, 
    ground_truth_files, 
    previously_assigned_file
):
    """
    Merges wide-format ground truth (from ground_truth_files) with previously assigned
    feedback (wide-format) from 'previously_assigned_file'.
    
    For each requirement in 'previously_assigned_file', picks ONE random piece of feedback
    (if available) and adds it to the ground truth. Returns the new ground truth path.
    """
    # 1) Combine your ground-truth files (axis=1) into one wide DataFrame
    gt_df_list = []
    for g in ground_truth_files:
        gdf = pd.read_excel(g, header=None)
        gt_df_list.append(gdf)
    # Concatenate “wide” along columns
    merged_gt_df = pd.concat(gt_df_list, axis=1, ignore_index=True)

    # 2) Read previously assigned feedback
    prev_df = pd.read_excel(previously_assigned_file, header=None)

    # Build dictionary from ground truth: { req_id : set(feedback_ids) }
    gt_dict = {}
    for col_idx in range(merged_gt_df.shape[1]):
        req_id = merged_gt_df.iloc[0, col_idx]
        if pd.isna(req_id):
            continue
        fb_set = set(merged_gt_df.iloc[1:, col_idx].dropna().tolist())
        gt_dict[req_id] = fb_set

    # Build dictionary from previously assigned: { req_id : set(feedback_ids) }
    prev_dict = {}
    for col_idx in range(prev_df.shape[1]):
        req_id = prev_df.iloc[0, col_idx]
        if pd.isna(req_id):
            continue
        fb_set = set(prev_df.iloc[1:, col_idx].dropna().tolist())
        prev_dict[req_id] = fb_set

    # 3) For each req_id in prev_dict, pick exactly ONE random feedback from assigned_fb_set
    for req_id, assigned_fb_set in prev_dict.items():
        if req_id not in gt_dict:
            gt_dict[req_id] = set()
        if assigned_fb_set:  # not empty
            one_fb = random.choice(list(assigned_fb_set))
            gt_dict[req_id].add(one_fb)

    # 4) Convert back to wide format
    all_req_ids = list(gt_dict.keys())
    max_count = max(len(gt_dict[r]) for r in gt_dict) if gt_dict else 0
    new_gt = pd.DataFrame(index=range(max_count+1), columns=range(len(all_req_ids)))

    for c, rid in enumerate(all_req_ids):
        new_gt.iat[0, c] = rid
        fb_list = list(gt_dict[rid])
        for r, fb_id in enumerate(fb_list, start=1):
            new_gt.iat[r, c] = fb_id

    # 5) Save to a temp XLSX
    out_dir = Path("./ongoing_experiments/temp_experiments_kfold")
    out_dir.mkdir(exist_ok=True)
    merged_gt_path = out_dir / "expanded_ground_truth_exp14.xlsx"
    new_gt.to_excel(merged_gt_path, header=False, index=False)

    return feedback_files, requirements_files, str(merged_gt_path)



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
                log_dir="./logs_kfold",
                results_dir="./kfold_results",
                exp_name="FeReRe_KFold_Combined",
                exp_id=exp["exp_id"]
            )
        
        # Special experiment 14
        else:
            # We'll accumulate metrics from 5 merges
            folds_metrics_all = []

            # We will do 5 merges, each time using "classified_feedback_requirements_exp_10_fold_{i}"
            for i in range(1, 6):
                previously_assigned_file = (
                    f"/nfs/home/vthakur_paech/kfold_results/"
                    f"classified_feedback_requirements_exp_10_fold_{i}.xlsx"
                )

                # 1) Incorporate that fold's previously assigned feedback
                fb_files, req_files, merged_gt_path = incorporate_previously_assigned_feedback_kfold(
                    feedback_files,
                    requirements_files,
                    ground_truth_files,
                    previously_assigned_file
                )

                # 2) Now run a brand-new 5-fold cross validation with that merged ground truth
                fold_metrics = train_eval_kfold_combined(
                    feedback_files=fb_files,
                    requirements_files=req_files,
                    ground_truth_files=[merged_gt_path],  # pass only the merged file
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
                # fold_metrics is a list of dicts, one per fold in train_eval_kfold_combined
                #   e.g. [{fold: 1, precision: p1, recall: r1, f2: f21, ...}, ...]

                # 3) Compute average across these 5 folds
                if len(fold_metrics) > 0:
                    avg_prec = sum(m["precision"] for m in fold_metrics) / len(fold_metrics)
                    avg_rec  = sum(m["recall"] for m in fold_metrics) / len(fold_metrics)
                    avg_f2   = sum(m["f2"] for m in fold_metrics) / len(fold_metrics)
                    avg_as   = sum(m["avg_assign"] for m in fold_metrics) / len(fold_metrics)
                    folds_metrics_all.append((avg_prec, avg_rec, avg_f2, avg_as))

            # After the loop, we have 5 sets of average metrics from each “merge + CV run”
            # Let's average them again so we get final metrics for Exp 14
            if folds_metrics_all:
                final_prec = sum(m[0] for m in folds_metrics_all) / len(folds_metrics_all)
                final_rec  = sum(m[1] for m in folds_metrics_all) / len(folds_metrics_all)
                final_f2   = sum(m[2] for m in folds_metrics_all) / len(folds_metrics_all)
                final_as   = sum(m[3] for m in folds_metrics_all) / len(folds_metrics_all)
                
                print("\n[EXPERIMENT 14] Final AVERAGE across all 5 merges of previous folds:")
                print(f"  Precision = {final_prec:.4f}")
                print(f"  Recall    = {final_rec:.4f}")
                print(f"  F2        = {final_f2:.4f}")
                print(f"  AvgAssigned = {final_as:.2f}")
                
                # Optionally log in MLflow:
                mlflow.set_experiment("FeReRe_KFold_Combined_Experiments")
                with mlflow.start_run(run_name="Experiment_14_FinalAverage"):
                    mlflow.log_param("exp_id", 14)
                    mlflow.log_metric("avg_precision", final_prec)
                    mlflow.log_metric("avg_recall", final_rec)
                    mlflow.log_metric("avg_f2", final_f2)
                    mlflow.log_metric("avg_assigned", final_as)
    
    print("\n[ALL EXPERIMENTS COMPLETED]\n")
if __name__ == "__main__":
    run_all_experiments()
