import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
import mlflow
import nltk
from nltk.tokenize import sent_tokenize
import random

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
feedback_df = pd.read_excel('../data/sbert/AppReviews.xlsx', header=None)
requirements_df = pd.read_excel('../data/sbert/jira_issues_noprefix.xlsx', header=None)
ground_truth_df = pd.read_excel('../data/sbert/Komoot_Ground_Truth_ids_only.xlsx')

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

# Create positive samples
positive_samples = []

for req_id, feedback_ids in ground_truth.items():
    req_indices = requirements_df[requirements_df.iloc[:, 0] == req_id].index
    if not req_indices.empty:
        req_index = req_indices[0]
        for feedback_id in feedback_ids:
            feedback_indices = feedback_df[feedback_df.iloc[:, 0] == feedback_id].index
            if not feedback_indices.empty:
                feedback_index = feedback_indices[0]
                positive_samples.append((req_index, feedback_index))

# Create negative samples by pairing randomly
negative_samples = []
while len(negative_samples) < len(positive_samples):
    req_index = random.choice(requirements_df.index)
    feedback_index = random.choice(feedback_df.index)
    if (req_index, feedback_index) not in positive_samples:
        negative_samples.append((req_index, feedback_index))

# Combine positive and negative samples and create labels
all_samples = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

# Create feature matrix
X = np.array([np.concatenate((requirements_padded[req], feedback_padded[fb])) for req, fb in all_samples])
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Bi-LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=2*max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPool1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Calculate avg_feedback_per_requirement based on the model predictions
req_feedback_counts = {req_id: 0 for req_id in requirements_df.iloc[:, 0]}

for i, (req_index, fb_index) in enumerate(all_samples[len(X_train):]):
    if y_pred[i] == 1:  # Consider only predicted positive pairs
        req_id = requirements_df.iloc[req_index, 0]
        req_feedback_counts[req_id] += 1

avg_feedback_per_requirement = np.mean(list(req_feedback_counts.values()))

print(f'Precision: {precision}, Recall: {recall}, F2 Score: {f2}, Avg Assign: {avg_feedback_per_requirement}')

# Save model
#model.save('bilstm_model.h5')

# Log metrics to MLFlow
'''
mlflow.set_experiment("Requirements-Feedback-Relation")
with mlflow.start_run():
    mlflow.log_param('epochs', 10)
    mlflow.log_param('batch_size', 32)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f2_score', f2)
    mlflow.log_metric('avg_feedback_per_requirement', avg_feedback_per_requirement)

mlflow.end_run()
'''
# Save results to Excel
results = {
    'Precision': [precision],
    'Recall': [recall],
    'F2 Score': [f2]
}
results_df = pd.DataFrame(results)
results_df.to_excel('../data/bilstm/bilstm_results.xlsx', index=False)

# Initialize dictionary to store results
predicted_results = {req_id: [] for req_id in requirements_df.iloc[:, 0]}

# Fill the dictionary with predicted feedback IDs
for i, (req_index, fb_index) in enumerate(all_samples[len(X_train):]):
    if y_pred[i] == 1:  # Consider only predicted positive pairs
        req_id = requirements_df.iloc[req_index, 0]
        fb_id = feedback_df.iloc[fb_index, 0]
        predicted_results[req_id].append(fb_id)

# Convert dictionary to DataFrame
max_len = max(len(v) for v in predicted_results.values())  # Find the maximum number of feedback per requirement

# Pad the feedback IDs with empty strings so all columns have the same length
for req_id in predicted_results:
    predicted_results[req_id] += [''] * (max_len - len(predicted_results[req_id]))

results_df = pd.DataFrame(predicted_results)

# Save the results to an Excel file
results_df.to_excel('../data/bilstm/classified_feedback_requirements.xlsx', index=False)

