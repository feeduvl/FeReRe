import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score
import mlflow

def load_excel(file_path):
    return pd.read_excel(file_path, header=None)

def compute_metrics(classifier_df, ground_truth_df):
    classifier_ids = classifier_df.iloc[0].dropna().tolist()
    ground_truth_ids = ground_truth_df.iloc[0].dropna().tolist()

    classifier_dict = {}
    ground_truth_dict = {}

    for col in range(classifier_df.shape[1]):
        feedback = classifier_df.iloc[1:, col].dropna().tolist()
        classifier_dict[classifier_ids[col]] = set(feedback)

    for col in range(ground_truth_df.shape[1]):
        feedback = ground_truth_df.iloc[1:, col].dropna().tolist()
        ground_truth_dict[ground_truth_ids[col]] = set(feedback)

    common_ids = set(classifier_dict.keys()) & set(ground_truth_dict.keys())

    y_true = []
    y_pred = []
    total_feedback_count = 0

    for req_id in common_ids:
        true_feedback = ground_truth_dict[req_id]
        pred_feedback = classifier_dict[req_id]
        all_feedback_ids = set(true_feedback) | set(pred_feedback)
        true_labels = [1 if fid in true_feedback else 0 for fid in all_feedback_ids]
        pred_labels = [1 if fid in pred_feedback else 0 for fid in all_feedback_ids]
        y_true.extend(true_labels)
        y_pred.extend(pred_labels)
        total_feedback_count += len(pred_feedback)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    avg_feedback_per_req = total_feedback_count / len(common_ids) if common_ids else 0

    return precision, recall, f2, avg_feedback_per_req

def run_eval(classifier, ground_truth, run_name=None):
    classifier_df = load_excel(classifier)
    ground_truth_df = load_excel(ground_truth)

    precision, recall, f2, avg_feedback_per_req = compute_metrics(classifier_df, ground_truth_df)

    mlflow.set_experiment("FeReRe_KFold_Experiments")
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_param("classifier_file", classifier)
        mlflow.log_param("ground_truth_file", ground_truth)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f2_score", f2)
        mlflow.log_metric("avg_assign", avg_feedback_per_req)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2 Score: {f2:.2f}')
    print(f'Average Feedback per Requirement: {avg_feedback_per_req:.2f}')
