import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, fbeta_score
import mlflow

# Load data
feedback_df = pd.read_excel('data/feedback.xlsx')
requirements_df = pd.read_excel('data/jira_issues_noprefix.xlsx')
ground_truth_df = pd.read_excel('data/Ground_Truth.xlsx')

# Extract text and IDs
feedback_ids = feedback_df.iloc[:, 0].tolist()
feedback_texts = feedback_df.iloc[:, 1].tolist()
requirement_ids = requirements_df.iloc[:, 0].tolist()
requirement_texts = requirements_df.iloc[:, 1].tolist()

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()
feedback_tfidf = vectorizer.fit_transform(feedback_texts)
requirements_tfidf = vectorizer.transform(requirement_texts)

# Prepare ground truth dictionary
ground_truth = {}
for col in ground_truth_df.columns:
    ground_truth[col] = ground_truth_df[col].dropna().tolist()[1:]

# Initialize results list
results = []

# Set initial threshold
threshold = 0.0

# Start MLFlow run
mlflow.start_run()
mlflow.set_experiment("Requirements-Feedback-Relation")
# Loop through thresholds using a while loop
while threshold <= 0.1:
    matching_pairs = []

    # Calculate cosine similarity and relate feedback to requirements
    for i, feedback_vector in enumerate(feedback_tfidf):
        for j, requirement_vector in enumerate(requirements_tfidf):
            similarity = cosine_similarity(feedback_vector, requirement_vector)[0][0]
            if similarity > threshold:
                matching_pairs.append((feedback_ids[i], requirement_ids[j]))

    # Prepare predictions dictionary
    predictions = {}
    for feedback_id, requirement_id in matching_pairs:
        if requirement_id not in predictions:
            predictions[requirement_id] = []
        predictions[requirement_id].append(feedback_id)

    # Flatten ground truth and predictions for metric calculation
    y_true = []
    y_pred = []
    for requirement_id in ground_truth:
        true_feedback_ids = set(ground_truth[requirement_id])
        pred_feedback_ids = set(predictions.get(requirement_id, []))
        for feedback_id in true_feedback_ids.union(pred_feedback_ids):
            y_true.append(feedback_id in true_feedback_ids)
            y_pred.append(feedback_id in pred_feedback_ids)

    # Calculate precision, recall, and F2 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)

    # Calculate average number of feedback assigned to each requirement
    avg_feedback_per_requirement = sum(len(feedback_ids) for feedback_ids in predictions.values()) / len(predictions) if predictions else 0

    # Store results
    results.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F2 Score': f2, 'Avg Assign': avg_feedback_per_requirement})

    # Log metrics to MLFlow
    mlflow.log_metric(f'{threshold}_Precision', precision)
    mlflow.log_metric(f'{threshold}_Recall', recall)
    mlflow.log_metric(f'{threshold}_F2', f2)
    mlflow.log_metric(f'{threshold}_AvgAssign', avg_feedback_per_requirement)


    # Increment threshold
    threshold += 0.01

# End MLFlow run
mlflow.end_run()

# Save results to Excel
results_df = pd.DataFrame(results)
results_df.to_excel('data/tfidf/threshold_metrics.xlsx', index=False)