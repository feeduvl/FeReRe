# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from nltk.tokenize import sent_tokenize
import random

def train_and_eval(feedback,requirements,ground_truth):
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

    # Flatten the lists of sentences
    feedback_sentences_flat = [sentence for sublist in feedback_sentences for sentence in sublist]
    requirements_sentences_flat = [sentence for sublist in requirements_sentences for sentence in sublist]

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(feedback_sentences_flat + requirements_sentences_flat)
    vocab_size = len(tokenizer.word_index) + 1

    feedback_sequences = tokenizer.texts_to_sequences(feedback_sentences_flat)
    requirements_sequences = tokenizer.texts_to_sequences(requirements_sentences_flat)

    max_length = max(max(len(seq) for seq in feedback_sequences), max(len(seq) for seq in requirements_sequences))
    feedback_padded = pad_sequences(feedback_sequences, maxlen=max_length, padding='post')
    requirements_padded = pad_sequences(requirements_sequences, maxlen=max_length, padding='post')

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

    # Tokenize and pad the sentence pairs
    all_sample_sequences = tokenizer.texts_to_sequences([req + ' ' + fb for req, fb in all_samples])
    all_sample_padded = pad_sequences(all_sample_sequences, maxlen=2*max_length, padding='post')

    # Create feature matrix
    X = np.array(all_sample_padded)
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build Bi-LSTM model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=2*max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),

    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=6, batch_size=64, validation_split=0.2)

    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    # Mapping original pairs to train/test sets
    all_pairs = positive_pairs + negative_pairs
    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(all_pairs)), y, test_size=0.2, random_state=42
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
    results_df.to_excel('../data/bilstm/bilstm_results.xlsx', index=False)

    # Convert dictionary to DataFrame
    max_len = max(len(v) for v in predicted_results.values())  # Find the maximum number of feedback per requirement

    # Pad the feedback IDs with empty strings so all columns have the same length
    for req_id in predicted_results:
        predicted_results[req_id] += [''] * (max_len - len(predicted_results[req_id]))

    results_df = pd.DataFrame(predicted_results)

    # Save the results to an Excel file
    results_df.to_excel('../data/bilstm/classified_feedback_requirements.xlsx', index=False)

train_and_eval("../data/smartage/SmartAgeSV_Feedback.xlsx","../data/smartage/SV_issues.xlsx","../data/smartage/SmartAgeSV_GT_formatted.xlsx")
#train_and_eval("../data/smartage/SmartAgeSF_Feedback.xlsx","../data/smartage/SF_issues.xlsx","../data/smartage/SmartAgeSF_GT_formatted.xlsx")
#train_and_eval("../data/sbert/AppReviews.xlsx","../data/sbert/jira_issues_noprefix.xlsx","../data/sbert/Komoot_Ground_Truth_ids_only.xlsx")