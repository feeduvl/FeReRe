import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score
import mlflow

# Load the Excel files
def load_excel(file_path):
    return pd.read_excel(file_path, header=None)

def compute_metrics(classifier_df, ground_truth_df):
    # Extract the IDs and feedback
    classifier_ids = classifier_df.iloc[0].dropna().tolist()
    ground_truth_ids = ground_truth_df.iloc[0].dropna().tolist()

    # Initialize dictionaries to store feedback for each ID
    classifier_dict = {}
    ground_truth_dict = {}

    # Populate classifier_dict
    for col in range(classifier_df.shape[1]):
        feedback = classifier_df.iloc[1:, col].dropna().tolist()  # Drop NaNs and convert to a list
        classifier_dict[classifier_ids[col]] = set(feedback)  # Store as a set

    # Populate ground_truth_dict
    for col in range(ground_truth_df.shape[1]):
        feedback = ground_truth_df.iloc[1:, col].dropna().tolist()  # Drop NaNs and convert to a list
        ground_truth_dict[ground_truth_ids[col]] = set(feedback)  # Store as a set

    # Find common IDs
    common_ids = set(classifier_dict.keys()) & set(ground_truth_dict.keys())

    # Initialize lists to store true and predicted labels
    y_true = []
    y_pred = []
    total_feedback_count = 0

    for req_id in common_ids:
        true_feedback = ground_truth_dict[req_id]
        pred_feedback = classifier_dict[req_id]
        # Create binary labels for feedback
        all_feedback_ids = set(true_feedback) | set(pred_feedback)
        true_labels = [1 if fid in true_feedback else 0 for fid in all_feedback_ids]
        pred_labels = [1 if fid in pred_feedback else 0 for fid in all_feedback_ids]
        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

        # Track the number of feedback assigned per requirement
        total_feedback_count += len(pred_feedback)

    # Calculate precision, recall, and F2 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2.0)

    # Calculate average number of feedback assigned per requirement
    avg_feedback_per_req = total_feedback_count / len(common_ids) if common_ids else 0

    return precision, recall, f2, avg_feedback_per_req

def run_eval(classifier, ground_truth):
    # File paths
    classifier_file_path = classifier
    ground_truth_file_path = ground_truth

    # Load data
    classifier_df = load_excel(classifier_file_path)
    ground_truth_df = load_excel(ground_truth_file_path)

    # Compute metrics
    precision, recall, f2, avg_feedback_per_req = compute_metrics(classifier_df, ground_truth_df)
    # Log metrics in MLflow
    mlflow.set_experiment("FinetunedFeReRe")
    with mlflow.start_run():
        mlflow.log_param("classifier_file", classifier_file_path)
        mlflow.log_param("ground_truth_file", ground_truth_file_path)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f2_score", f2)
        mlflow.log_metric("avg_assign", avg_feedback_per_req)
    mlflow.end_run()
    # Print results
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2 Score: {f2:.2f}')
    print(f'Average Feedback per Requirement: {avg_feedback_per_req:.2f}')


def run_multiple_evals(names, classifier_files, ground_truth_files):
    if len(classifier_files) != len(ground_truth_files):
        raise ValueError("The number of classifier files and ground truth files must be the same")

    overall_precision = 0
    overall_recall = 0
    overall_f2 = 0
    overall_avg_feedback = 0
    n = len(classifier_files)

    mlflow.set_experiment("FinetunedFeReRe")
    mlflow.start_run()

    for name, classifier, ground_truth in zip(names, classifier_files, ground_truth_files):
        classifier_df = load_excel(classifier)
        ground_truth_df = load_excel(ground_truth)
        print(f"Evaluating {name}")
        precision, recall, f2, avg_feedback_per_req = compute_metrics(classifier_df, ground_truth_df)
        mlflow.log_metric(f"precision_{name}", precision)
        mlflow.log_metric(f"recall_{name}", recall)
        mlflow.log_metric(f"f2_score_{name}", f2)
        mlflow.log_metric(f"avg_assign_{name}", avg_feedback_per_req)

        overall_precision += precision
        overall_recall += recall
        overall_f2 += f2
        overall_avg_feedback += avg_feedback_per_req

    # Calculate overall averages
    overall_precision /= n
    overall_recall /= n
    overall_f2 /= n
    overall_avg_feedback /= n

    print("\nOverall Performance:")
    print(f'Average Precision: {overall_precision:.2f}')
    print(f'Average Recall: {overall_recall:.2f}')
    print(f'Average F2 Score: {overall_f2:.2f}')
    print(f'Average Feedback per Requirement: {overall_avg_feedback:.2f}')

    mlflow.log_metric("precision", overall_precision)
    mlflow.log_metric("recall", overall_recall)
    mlflow.log_metric("f2_score", overall_f2)
    mlflow.log_metric("avg_assign", overall_avg_feedback)
    mlflow.end_run()

#BERT
#run_eval('../data/finetuneBERT/classified_feedback_requirements.xlsx', '../data/finetuneBERT/test_ground_truth.xlsx')

#Bert kfold
#classifier_files = ["../data/finetuneBERT/kfold/classified_feedback_requirements_1.xlsx", "../data/finetuneBERT/kfold/classified_feedback_requirements_2.xlsx", "../data/finetuneBERT/kfold/classified_feedback_requirements_3.xlsx", "../data/finetuneBERT/kfold/classified_feedback_requirements_4.xlsx", "../data/finetuneBERT/kfold/classified_feedback_requirements_5.xlsx"]
#ground_truth_files = ["../data/finetuneBERT/kfold/test_ground_truth_1.xlsx", "../data/finetuneBERT/kfold/test_ground_truth_2.xlsx", "../data/finetuneBERT/kfold/test_ground_truth_3.xlsx", "../data/finetuneBERT/kfold/test_ground_truth_4.xlsx", "../data/finetuneBERT/kfold/test_ground_truth_5.xlsx"]
#names=["1", "2", "3", "4", "5"]
#run_multiple_evals(names, classifier_files, ground_truth_files)

#Bert multidata
#classifier_files = ["../data/finetuneBERT/multidata/classified_feedback_requirements_0.xlsx", "../data/finetuneBERT/multidata/classified_feedback_requirements_1.xlsx", "../data/finetuneBERT/multidata/classified_feedback_requirements_2.xlsx"]
#ground_truth_files = ["../data/finetuneBERT/multidata/test_ground_truth_0.xlsx", "../data/finetuneBERT/multidata/test_ground_truth_1.xlsx", "../data/finetuneBERT/multidata/test_ground_truth_2.xlsx"]
#names=["Komoot", "SV", "SF"]
#run_multiple_evals(names, classifier_files, ground_truth_files)

#Bert splittesttrain
#run_eval('../data/finetuneBERT/splittesttrain/classified_feedback_requirements_0.xlsx', '../data/finetuneBERT/splittesttrain/test_ground_truth_0.xlsx')

#Bert combineddata
run_eval('../data/finetuneBERT/combineddata/classified_feedback_requirements.xlsx', '../data/finetuneBERT/combineddata/test_ground_truth.xlsx')