import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import torch
import mlflow

def cross_encoder():
    # Download the punkt tokenizer for sentence splitting
    #nltk.download('punkt')

    # Check if GPU is available and set device
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    device = 'cuda' if cuda_available else 'cpu'

    # Load pre-trained Cross-Encoder model and move it to the appropriate device
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    # Read feedback and requirements from Excel files
    feedback_df = pd.read_excel('data/sbert/AppReviews.xlsx')
    requirements_df = pd.read_excel('data/sbert/jira_issues_noprefix.xlsx')

    # Extract text
    feedback = feedback_df.iloc[:, 1].tolist()
    requirements = requirements_df.iloc[:, 1].tolist()

    # Read ground truth
    ground_truth_df = pd.read_excel('data/sbert/Komoot_Ground_Truth_ids_only.xlsx')

    # Prepare ground truth dictionary
    ground_truth = {}
    for col in ground_truth_df.columns:
        ground_truth[col] = ground_truth_df[col].dropna().tolist()[1:]

    # Initialize results list
    results = []
    print("Calculating scores...")
    # Compute similarity scores once
    similarity_scores = []
    for feedback_id, feedback_text in zip(feedback_df.iloc[:, 0], feedback):
        feedback_sentences = nltk.sent_tokenize(feedback_text)
        for i, requirement_text in enumerate(requirements):
            requirement_sentences = nltk.sent_tokenize(requirement_text)
            max_score = 0
            for req_sentence in requirement_sentences:
                pairs = [[req_sentence, feedback_sentence] for feedback_sentence in feedback_sentences]
                scores = model.predict(pairs, activation_fct=torch.nn.Sigmoid())
                max_score = max(max_score, max(scores))
            similarity_scores.append((feedback_id, requirements_df.iloc[i, 0], max_score))
    print("Evaluating...")
    mlflow.set_experiment("Requirements-Feedback-Relation")
    # Start MLFlow run
    with mlflow.start_run():
        # Loop through thresholds
        for threshold in [i * 0.1 for i in range(11)]:
            print(threshold)
            matching_pairs = [(feedback_id, requirement_id) for feedback_id, requirement_id, score in similarity_scores if score > threshold]

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

            # Calculate precision, recall, and F1 score
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f2 = fbeta_score(y_true, y_pred, beta=2)

            # Calculate average number of feedback assigned to each requirement
            avg_feedback_per_requirement = sum(len(feedback_ids) for feedback_ids in predictions.values()) / len(predictions) if predictions else 0

            # Store results
            results.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F2 Score': f2, 'Avg Feedback per Requirement': avg_feedback_per_requirement})

            # Log metrics and parameters to MLFlow
            mlflow.log_metric(f'{threshold}Precision', precision)
            mlflow.log_metric(f'{threshold}Recall', recall)
            mlflow.log_metric(f'{threshold}F2', f2)
            mlflow.log_metric(f'{threshold}AvgRel', avg_feedback_per_requirement)

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel('data/sbert/threshold_metrics.xlsx', index=False)
    mlflow.end_run()

def cross_encoder_tfidf_weights():
    # Download the punkt tokenizer for sentence splitting
    nltk.download('punkt')

    # Check if GPU is available and set device
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    device = 'cuda' if cuda_available else 'cpu'

    # Load pre-trained Cross-Encoder model and move it to the appropriate device
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    # Read feedback and requirements from Excel files
    feedback_df = pd.read_excel('data/sbert/AppReviews.xlsx')
    requirements_df = pd.read_excel('data/sbert/jira_issues_noprefix.xlsx')

    # Extract text
    feedback = feedback_df.iloc[:, 1].tolist()
    requirements = requirements_df.iloc[:, 1].tolist()

    # Combine all text for TF-IDF vectorizer
    combined_text = feedback + requirements

    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined_text)

    # Read ground truth
    ground_truth_df = pd.read_excel('data/sbert/Komoot_Ground_Truth_ids_only.xlsx')

    # Prepare ground truth dictionary
    ground_truth = {}
    for col in ground_truth_df.columns:
        ground_truth[col] = ground_truth_df[col].dropna().tolist()[1:]

    # Initialize results list
    results = []
    print("Calculating scores...")

    # Compute similarity scores once
    similarity_scores = []
    for feedback_id, feedback_text in zip(feedback_df.iloc[:, 0], feedback):
        feedback_sentences = nltk.sent_tokenize(feedback_text)
        for i, requirement_text in enumerate(requirements):
            requirement_sentences = nltk.sent_tokenize(requirement_text)
            max_score = 0
            for req_sentence in requirement_sentences:
                for feedback_sentence in feedback_sentences:
                    pairs = [[req_sentence, feedback_sentence]]
                    scores = model.predict(pairs, activation_fct=torch.nn.Sigmoid())

                    # Calculate TF-IDF weighted score
                    tfidf_scores = vectorizer.transform([req_sentence, feedback_sentence])
                    weighted_scores = scores * tfidf_scores.toarray().sum(axis=1)
                    max_score = max(max_score, max(weighted_scores))
            similarity_scores.append((feedback_id, requirements_df.iloc[i, 0], max_score))

    print("Evaluating...")
    mlflow.set_experiment("Requirements-Feedback-Relation")

    # Start MLFlow run
    with mlflow.start_run():
        # Loop through thresholds
        for threshold in [i * 0.1 for i in range(11)]:
            print(threshold)
            matching_pairs = [(feedback_id, requirement_id) for feedback_id, requirement_id, score in similarity_scores if score > threshold]

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

            # Calculate precision, recall, and F1 score
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f2 = fbeta_score(y_true, y_pred, beta=2)

            # Calculate average number of feedback assigned to each requirement
            avg_feedback_per_requirement = sum(len(feedback_ids) for feedback_ids in predictions.values()) / len(predictions) if predictions else 0

            # Store results
            results.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F2 Score': f2, 'Avg Feedback per Requirement': avg_feedback_per_requirement})

            # Log metrics and parameters to MLFlow
            mlflow.log_metric(f'{threshold}Precision', precision)
            mlflow.log_metric(f'{threshold}Recall', recall)
            mlflow.log_metric(f'{threshold}F2', f2)
            mlflow.log_metric(f'{threshold}AvgRel', avg_feedback_per_requirement)

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel('data/sbert/threshold_metrics.xlsx', index=False)
    mlflow.end_run()

def sentence_transformer():
    # Download the punkt tokenizer for sentence splitting
    nltk.download('punkt')

    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Read feedback and requirements from Excel files
    feedback_df = pd.read_excel('data/sbert/AppReviews.xlsx')
    requirements_df = pd.read_excel('data/sbert/jira_issues_noprefix.xlsx')

    # Extract text
    feedback = feedback_df.iloc[:, 1].tolist()
    requirements = requirements_df.iloc[:, 1].tolist()

    # Compute embeddings for requirements
    requirements_embeddings = model.encode(requirements, convert_to_tensor=True)

    # Read ground truth
    ground_truth_df = pd.read_excel('data/sbert/Komoot_Ground_Truth_ids_only.xlsx')

    # Prepare ground truth dictionary
    ground_truth = {}
    for col in ground_truth_df.columns:
        ground_truth[col] = ground_truth_df[col].dropna().tolist()[1:]

    # Initialize results list
    results = []

    # Loop through thresholds
    for threshold in [i * 0.05 for i in range(21)]:
        print(threshold)
        matching_pairs = []

        # Process each feedback
        for feedback_id, feedback_text in zip(feedback_df.iloc[:, 0], feedback):
            # Split feedback into sentences using nltk
            sentences = nltk.sent_tokenize(feedback_text)
            # Compute embeddings for each sentence
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

            # Calculate cosine similarity between each requirement and each sentence of feedback
            for i, requirement_embedding in enumerate(requirements_embeddings):
                cosine_scores = util.pytorch_cos_sim(requirement_embedding, sentence_embeddings)
                max_score = cosine_scores.max().item()
                if max_score > threshold:
                    matching_pairs.append((feedback_id, requirements_df.iloc[i, 0], max_score))

        # Prepare predictions dictionary
        predictions = {}
        for feedback_id, requirement_id, _ in matching_pairs:
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

        # Calculate precision, recall, and F1 score
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2)

        # Calculate average number of feedback assigned to each requirement
        avg_feedback_per_requirement = sum(len(feedback_ids) for feedback_ids in predictions.values()) / len(predictions) if predictions else 0

        # Store results
        results.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F2 Score': f2, 'Avg Feedback per Requirement': avg_feedback_per_requirement})

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel('data/sbert/threshold_metrics.xlsx', index=False)

cross_encoder_tfidf_weights()