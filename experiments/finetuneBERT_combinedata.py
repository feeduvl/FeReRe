import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from nltk.tokenize import sent_tokenize
import nltk
import random

def load_and_combine_files(file_list, concat_axis=0):
    combined_df = pd.read_excel(file_list[0], header=None)

    for file in file_list[1:]:
        df = pd.read_excel(file, header=None)
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], axis=concat_axis, ignore_index=True)
    return combined_df

def train_and_eval(feedback_files, requirements_files, ground_truth_files, epochs=1):
    nltk.download('punkt')
    # Ensure TensorFlow uses GPU
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Load data
    feedback_df = load_and_combine_files(feedback_files)
    requirements_df = load_and_combine_files(requirements_files)
    ground_truth_df = load_and_combine_files(ground_truth_files, concat_axis=1)

    # Extract text
    feedback = feedback_df.iloc[:, 1].tolist()
    requirements = requirements_df.iloc[:, 1].tolist()

    # Prepare ground truth dictionary
    ground_truth = {}
    for col in ground_truth_df.columns:
        key = ground_truth_df[col].iloc[0]  # First value in the column as the key
        values = ground_truth_df[col].iloc[1:].dropna().tolist()  # Remaining values as the list
        ground_truth[key] = values

    # Split feedback and requirements into sentences
    feedback_sentences = [sent_tokenize(text) for text in feedback]
    requirements_sentences = [sent_tokenize(text) for text in requirements]

    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

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

    # Tokenize and encode the sentence pairs using BERT tokenizer
    max_length=128# Adjust max_length as needed
    encoded_inputs = tokenizer(
        [req for req, fb in all_samples],
        [fb for req, fb in all_samples],
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    input_ids = np.array(encoded_inputs['input_ids'])
    attention_mask = np.array(encoded_inputs['attention_mask'])

    # Split data
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42
    )
    pd.DataFrame(y_train).to_excel('../data/finetuneBERT/combineddata/trainlabel.xlsx', index=False)
    pd.DataFrame(y_test).to_excel('../data/finetuneBERT/combineddata/testlabel.xlsx', index=False)
    # Build BERT-based model
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    bert_output = bert_model(input_ids_layer, attention_mask=attention_mask_layer)[1]  # Get pooled output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)

    model = tf.keras.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)

    learning_rate= 2e-5

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        [X_train_ids, X_train_mask],
        np.array(y_train),
        epochs=epochs,  # BERT models typically converge faster, so fewer epochs might be sufficient
        batch_size=32,
        validation_split=0.2
    )

    # Save model
    model.save('../data/finetuneBERT/combineddata/trained_model')

    # Evaluate model
    prediction= model.predict([X_test_ids, X_test_mask])
    pd.DataFrame(prediction).to_excel('../data/finetuneBERT/combineddata/predictions.xlsx', index=False)
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
    results_df.to_excel('../data/finetuneBERT/combineddata/bert_results.xlsx', index=False)

    # Convert dictionary to DataFrame
    max_len = max(len(v) for v in predicted_results.values())  # Find the maximum number of feedback per requirement

    # Pad the feedback IDs with empty strings so all columns have the same length
    for req_id in predicted_results:
        predicted_results[req_id] += [''] * (max_len - len(predicted_results[req_id]))

    results_df = pd.DataFrame(predicted_results)

    # Save the results to an Excel file
    results_df.to_excel('../data/finetuneBERT/combineddata/classified_feedback_requirements.xlsx', index=False)

    # Save ground truth for the test split
    # Extract indices of the test set pairs
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
    test_ground_truth_df.to_excel('../data/finetuneBERT/combineddata/test_ground_truth.xlsx', index=False)

feedback_files = ["../data/smartage/SmartAgeSV_Feedback.xlsx", "../data/smartage/SmartAgeSF_Feedback.xlsx", "../data/komoot/AppReviews.xlsx", "../data/ReFeed/feedback.xlsx"]
requirements_files = ["../data/smartage/SV_issues.xlsx", "../data/smartage/SF_issues.xlsx", "../data/komoot/jira_issues_noprefix.xlsx", "../data/ReFeed/requirements.xlsx"]
ground_truth_files = ["../data/smartage/SmartAgeSV_GT_formatted.xlsx", "../data/smartage/SmartAgeSF_GT_formatted.xlsx", "../data/komoot/Komoot_Ground_Truth_ids_only.xlsx", "../data/ReFeed/refeed_gt.xlsx"]

train_and_eval(feedback_files, requirements_files, ground_truth_files, 4)
