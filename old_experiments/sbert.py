import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from nltk.tokenize import sent_tokenize
import nltk
import random

def train_and_eval(feedback, requirements, ground_truth, epochs=1):
    nltk.download('punkt')
    # Ensure TensorFlow uses GPU
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Load data
    feedback_df = pd.read_excel(feedback, header=None)
    requirements_df = pd.read_excel(requirements, header=None)
    ground_truth_df = pd.read_excel(ground_truth)

    # Extract text
    feedback = feedback_df.iloc[:, 1].tolist()
    requirements = requirements_df.iloc[:, 1].tolist()

    # Prepare ground truth dictionary
    ground_truth = {}
    for col in ground_truth_df.columns:
        ground_truth[col] = ground_truth_df[col].dropna().tolist()

    # Split feedback and requirements into sentences
    feedback_sentences = [sent_tokenize(text) for text in feedback]
    requirements_sentences = [sent_tokenize(text) for text in requirements]

    # Initialize SBERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can change the model as needed

    # Create positive samples at the sentence-pair level
    positive_samples = []
    positive_pairs = []

    for req_id, feedback_ids in ground_truth.items():
        req_indices = requirements_df[requirements_df.iloc[:, 0] == req_id].index
        if not req_indices.empty:
            req_index = req_indices[0]
            req_sentences = requirements_sentences[req_index]
            for feedback_id in feedback_ids:
                feedback_indices = feedback_df[feedback_df.iloc[:, 0] == feedback_id].index
                if not feedback_indices.empty:
                    feedback_index = feedback_indices[0]
                    feedback_sentences_list = feedback_sentences[feedback_index]
                    for req_sentence in req_sentences:
                        for fb_sentence in feedback_sentences_list:
                            positive_samples.append((req_sentence, fb_sentence))
                            positive_pairs.append((req_index, feedback_index))

    # Create negative samples by pairing randomly
    negative_samples = []
    negative_pairs = []

    while len(negative_samples) < len(positive_samples):
        req_index = random.choice(requirements_df.index)
        feedback_index = random.choice(feedback_df.index)
        req_sentences = requirements_sentences[req_index]
        feedback_sentences_list = feedback_sentences[feedback_index]
        if (req_index, feedback_index) not in positive_pairs:
            for req_sentence in req_sentences:
                for fb_sentence in feedback_sentences_list:
                    negative_samples.append((req_sentence, fb_sentence))
                    negative_pairs.append((req_index, feedback_index))

    # Combine positive and negative samples and create labels
    all_samples = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    # Encode the sentence pairs using SBERT model
    sentence_embeddings = sbert_model.encode([f"{req} [SEP] {fb}" for req, fb in all_samples])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sentence_embeddings, labels, test_size=0.2, random_state=42
    )

    # Build a simple classifier (logistic regression with one dense layer)
    input_layer = tf.keras.layers.Input(shape=(sentence_embeddings.shape[1],), dtype=tf.float32, name='sentence_embeddings')
    output = tf.keras.layers.Dense(1, activation='sigmoid')(input_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    learning_rate = 2e-5

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        np.array(X_train),
        np.array(y_train),
        epochs=epochs,  # SBERT models converge even faster, so fewer epochs might be sufficient
        batch_size=32,
        validation_split=0.2
    )

    # Evaluate model
    prediction = model.predict(np.array(X_test))
    pd.DataFrame(prediction).to_excel('../data/sbert/predictions.xlsx', index=False)
    y_pred = (prediction > 0.5).astype("int32")

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    # Mapping original pairs to train/test sets
    all_pairs = positive_pairs + negative_pairs
    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(all_pairs)), labels, test_size=0.2, random_state=42
    )

    # Aggregating predictions at the feedback-requirement level
    predicted_results = {req_id: [] for req_id in requirements_df.iloc[:, 0]}
    req_feedback_counts = {req_id: 0 for req_id in requirements_df.iloc[:, 0]}

    # Ensure y_pred is the same length as test_indices
    assert len(y_pred) == len(test_indices)

    for idx, y_pred_value in zip(test_indices, y_pred):
        req_index, fb_index = all_pairs[idx]

        if y_pred_value == 1:  # Consider only predicted positive pairs
            req_id = requirements_df.iloc[req_index, 0]
            fb_id = feedback_df.iloc[fb_index, 0]
            if fb_id not in predicted_results[req_id]:
                predicted_results[req_id].append(fb_id)
                req_feedback_counts[req_id] += 1

    avg_feedback_per_requirement = np.mean(list(req_feedback_counts.values()))

    print(f'Precision: {precision}, Recall: {recall}, F2 Score: {f2}, Avg Assign: {avg_feedback_per_requirement}')

    # Save results to Excel
    results = {
        'Precision': [precision],
        'Recall': [recall],
        'F2 Score': [f2],
        "Avg Assigned": [avg_feedback_per_requirement]
    }
    results_df = pd.DataFrame(results)
    results_df.to_excel('../data/sbert/sbert_results.xlsx', index=False)

    # Convert dictionary to DataFrame
    max_len = max(len(v) for v in predicted_results.values())  # Find the maximum number of feedback per requirement

    # Pad the feedback IDs with empty strings so all columns have the same length
    for req_id in predicted_results:
        predicted_results[req_id] += [''] * (max_len - len(predicted_results[req_id]))

    results_df = pd.DataFrame(predicted_results)

    # Save the results to an Excel file
    results_df.to_excel('../data/sbert/classified_feedback_requirements.xlsx', index=False)

    # Save ground truth for the test split
    test_pairs = [all_pairs[i] for i in test_indices]
    test_ground_truth = {req_id: [] for req_id in requirements_df.iloc[:, 0]}

    for req_index, fb_index in test_pairs:
        req_id = requirements_df.iloc[req_index, 0]
        fb_id = feedback_df.iloc[fb_index, 0]
        if fb_id in ground_truth.get(req_id, []) and fb_id not in test_ground_truth[req_id]:
            test_ground_truth[req_id].append(fb_id)

    max_len = max(len(v) for v in test_ground_truth.values())  # Find the maximum number of feedback per requirement

    # Pad the feedback IDs with empty strings so all columns have the same length
    for req_id in test_ground_truth:
        test_ground_truth[req_id] += [''] * (max_len - len(test_ground_truth[req_id]))

    test_ground_truth_df = pd.DataFrame(test_ground_truth)
    test_ground_truth_df.to_excel('../data/sbert/test_ground_truth.xlsx', index=False)



#train_and_eval("../data/smartage/SmartAgeSV_Feedback.xlsx", "../data/smartage/SV_issues.xlsx", "../data/smartage/SmartAgeSV_GT_formatted.xlsx", 3)
#train_and_eval("../data/smartage/SmartAgeSF_Feedback.xlsx", "../data/smartage/SF_issues.xlsx", "../data/smartage/SmartAgeSF_GT_formatted.xlsx", 3)
train_and_eval("../data/sbert/AppReviews.xlsx","../data/sbert/jira_issues_noprefix.xlsx","../data/sbert/Komoot_Ground_Truth_ids_only.xlsx", 500)

#train_and_eval("../data/ReFeed/feedback.xlsx", "../data/ReFeed/requirements.xlsx", "../data/ReFeed/refeed_gt.xlsx", 5)