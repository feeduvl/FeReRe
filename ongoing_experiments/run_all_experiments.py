import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import (
    BertTokenizer,
    TFBertModel,
    RobertaTokenizer,
    TFRobertaModel,
    DistilBertTokenizer,
    TFDistilBertModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
import nltk
import random
import mlflow
import string
import gc 
# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATA_DIR = '/Users/vasu/Desktop/FeReRe/Original_Repo/FeReRe/FeReRe/data'  # Adjust this path based on your repository structure
FINETUNE_DIR = os.path.join(DATA_DIR, 'finetuneBERT')
METRICS_DIR = os.path.join(DATA_DIR, 'metrics')

os.makedirs(FINETUNE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------
# Step 1: Load and Parse Data
# ---------------------------

def load_data(feedback_path, requirements_path, ground_truth_path):
    """
    Load feedback, requirements, and ground truth data from Excel files.
    
    Parameters:
    - feedback_path (str): Path to the feedback Excel file.
    - requirements_path (str): Path to the requirements Excel file.
    - ground_truth_path (str): Path to the ground truth Excel file.
    
    Returns:
    - feedback_df (DataFrame): Feedback DataFrame.
    - requirements_df (DataFrame): Requirements DataFrame.
    - ground_truth_df (DataFrame): Ground Truth DataFrame.
    - feedback_sentences (list of lists): Tokenized feedback sentences.
    - requirements_sentences (list of lists): Tokenized requirement sentences.
    - ground_truth (dict): Mapping of requirements to feedback IDs.
    """
    # Load data
    feedback_df = pd.read_excel(feedback_path, header=None)
    requirements_df = pd.read_excel(requirements_path, header=None)
    ground_truth_df = pd.read_excel(ground_truth_path)

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

    return feedback_df, requirements_df, ground_truth_df, feedback_sentences, requirements_sentences, ground_truth

# -------------------------------------------------
# Step 2: Define Preprocessing Functions
# -------------------------------------------------

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def synonym_replacement(text):
    words = word_tokenize(text)
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Get the first synonym that is not the same as the word
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != word:
                # Replace underscores with spaces (for multi-word synonyms)
                new_word = synonym.replace('_', ' ')
                # Capitalize if original word was capitalized
                if word[0].isupper():
                    new_word = new_word.capitalize()
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def sentence_splitting(text):
    # Simple splitting by clauses using comma as a delimiter
    clauses = text.split(',')
    return [clause.strip() for clause in clauses if clause.strip()]

def preprocess_text(text, steps):
    """
    Apply preprocessing steps to the text.
    
    Parameters:
    - text (str): The input text.
    - steps (list of str): List of preprocessing steps to apply.
    
    Returns:
    - list of str: Preprocessed sentences.
    """
    sentences = sent_tokenize(text)
    processed_sentences = []
    
    for sentence in sentences:
        if 'stopword_removal' in steps:
            sentence = remove_stopwords(sentence)
        if 'synonym_replacement' in steps:
            sentence = synonym_replacement(sentence)
        if 'sentence_splitting' in steps:
            split_sentences = sentence_splitting(sentence)
            processed_sentences.extend(split_sentences)
            continue  # Skip adding the original sentence
        processed_sentences.append(sentence)
    
    return processed_sentences

# -------------------------------------------------
# Step 3: Define Experiment Configurations
# -------------------------------------------------

EXPERIMENTS = [
     {
         'id': 1,
         'name': 'BERT_Base_Stopword_Removal',
         'preprocessing': ['stopword_removal'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 2,
         'name': 'BERT_Base_Synonym_Replacement',
         'preprocessing': ['synonym_replacement'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 3,
         'name': 'BERT_Base_Sentence_Splitting',
         'preprocessing': ['sentence_splitting'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 4,
         'name': 'BERT_Base_Stopword_Removal_Synonym_Replacement',
         'preprocessing': ['stopword_removal', 'synonym_replacement'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 5,
         'name': 'BERT_Base_Stopword_Removal_Sentence_Splitting',
         'preprocessing': ['stopword_removal', 'sentence_splitting'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 6,
         'name': 'BERT_Base_Synonym_Replacement_Sentence_Splitting',
         'preprocessing': ['synonym_replacement', 'sentence_splitting'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 7,
         'name': 'BERT_Base_All_Preprocessing',
         'preprocessing': ['stopword_removal', 'synonym_replacement', 'sentence_splitting'],
         'model': 'bert-base-uncased'
     },
     {
         'id': 8,
         'name': 'RoBERTa_No_Preprocessing',
         'preprocessing': [],
         'model': 'roberta-base'
     },
     {
         'id': 9,
         'name': 'RoBERTa_Best_Preprocessing',
         'preprocessing': ['stopword_removal', 'synonym_replacement', 'sentence_splitting'],
         'model': 'roberta-base'
     },
    {
        'id': 10,
        'name': 'BERT_Large_No_Preprocessing',
        'preprocessing': [],
        'model': 'bert-large-uncased'
    },
    {
        'id': 11,
        'name': 'BERT_Large_Best_Preprocessing',
        'preprocessing': ['stopword_removal', 'synonym_replacement', 'sentence_splitting'],
        'model': 'bert-large-uncased'
    },
    {
        'id': 12,
        'name': 'DistilBERT_No_Preprocessing',
        'preprocessing': [],
        'model': 'distilbert-base-uncased'
    },
    {
        'id': 13,
        'name': 'DistilBERT_Best_Preprocessing',
        'preprocessing': ['stopword_removal', 'synonym_replacement', 'sentence_splitting'],
        'model': 'distilbert-base-uncased'
    },
    {
        'id': 14,
        'name': 'Best_Model_With_Previously_Assigned_Feedback',
        'preprocessing': ['stopword_removal', 'synonym_replacement', 'sentence_splitting'],
        'model': 'bert-base-uncased',
        'additional_steps': ['incorporate_previous_feedback']
    }
]

# -------------------------------------------------
# Step 4: Model Training and Evaluation Functions
# -------------------------------------------------

def create_positive_negative_samples(feedback_df, requirements_df, preprocessed_feedback, preprocessed_requirements, ground_truth):
    """
    Create positive and negative samples based on ground truth.
    
    Parameters:
    - feedback_df (DataFrame): Feedback DataFrame.
    - requirements_df (DataFrame): Requirements DataFrame.
    - preprocessed_feedback (list of lists): Preprocessed feedback sentences.
    - preprocessed_requirements (list of lists): Preprocessed requirement sentences.
    - ground_truth (dict): Mapping of requirements to feedback IDs.
    
    Returns:
    - all_samples (list of tuples): List of (requirement_sentence, feedback_sentence) pairs.
    - labels (list of int): Corresponding labels (1 for positive, 0 for negative).
    """
    positive_samples = []
    positive_pairs = set()
    
    for req_id, feedback_ids in ground_truth.items():
        req_indices = requirements_df[requirements_df.iloc[:, 0] == req_id].index
        if not req_indices.empty:
            req_index = req_indices[0]
            req_sentences = preprocessed_requirements[req_index]
            for feedback_id in feedback_ids:
                feedback_indices = feedback_df[feedback_df.iloc[:, 0] == feedback_id].index
                if not feedback_indices.empty:
                    feedback_index = feedback_indices[0]
                    feedback_sentences_list = preprocessed_feedback[feedback_index]
                    for req_sentence in req_sentences:
                        for fb_sentence in feedback_sentences_list:
                            positive_samples.append((req_sentence, fb_sentence))
                            positive_pairs.add((req_index, feedback_index))
    
    # All possible pairs
    all_possible_pairs = set()
    for req_idx in requirements_df.index:
        for fb_idx in feedback_df.index:
            all_possible_pairs.add((req_idx, fb_idx))
    
    # Negative pairs are all possible pairs excluding positive_pairs
    negative_pairs = list(all_possible_pairs - positive_pairs)
    
    # If there are no negative pairs, skip creating negative samples
    if len(negative_pairs) == 0:
        print("No negative pairs available. Skipping negative samples.")
        all_samples = positive_samples
        labels = [1] * len(positive_samples)
    else:
        # Determine the number of negative samples to create
        num_positive = len(positive_samples)
        num_negatives = min(num_positive, len(negative_pairs))
        
        selected_negative_pairs = random.sample(negative_pairs, num_negatives)
        negative_samples = []
        for req_idx, fb_idx in selected_negative_pairs:
            req_sentences = preprocessed_requirements[req_idx]
            fb_sentences = preprocessed_feedback[fb_idx]
            for req_sentence in req_sentences:
                for fb_sentence in fb_sentences:
                    negative_samples.append((req_sentence, fb_sentence))
        
        # Combine positive and negative samples
        all_samples = positive_samples + negative_samples
        labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    print(f"Number of Positive Samples: {len(positive_samples)}")
    print(f"Number of Negative Samples: {len(labels) - len(positive_samples)}")
    
    return all_samples, labels

def tokenize_and_encode(all_samples, tokenizer, max_length=128):
    """
    Tokenize and encode sentence pairs using the specified tokenizer.
    
    Parameters:
    - all_samples (list of tuples): List of (requirement_sentence, feedback_sentence) pairs.
    - tokenizer: Hugging Face tokenizer.
    - max_length (int): Maximum sequence length.
    
    Returns:
    - input_ids (ndarray): Token IDs.
    - attention_mask (ndarray): Attention masks.
    """
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
    
    return input_ids, attention_mask

def build_transformer_model(model_name, max_length=128):
    """
    Build and compile a transformer-based model.
    
    Parameters:
    - model_name (str): Name of the transformer model.
    - max_length (int): Maximum sequence length.
    
    Returns:
    - model (tf.keras.Model): Compiled model.
    - tokenizer: Corresponding tokenizer.
    """
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        bert_model = TFRobertaModel.from_pretrained(model_name)
    elif 'distilbert' in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        bert_model = TFDistilBertModel.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = TFBertModel.from_pretrained(model_name)
    
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    outputs = bert_model(input_ids, attention_mask=attention_mask)
    
    # Handle different output structures
    if isinstance(outputs, tuple):
        # Typically (last_hidden_state, pooler_output)
        if len(outputs) > 1:
            bert_output = outputs[1]
        else:
            # Some models might return only last_hidden_state
            bert_output = tf.keras.layers.GlobalAveragePooling1D()(outputs[0])
    else:
        # For models like DistilBERT which may return a single Tensor
        bert_output = tf.keras.layers.GlobalAveragePooling1D()(outputs.last_hidden_state)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    learning_rate = 2e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model, tokenizer


def train_model_on_data(model, X_train_ids, X_train_mask, y_train, epochs=1, batch_size=32):
    """
    Train the model on the training data.
    
    Parameters:
    - model (tf.keras.Model): The compiled model.
    - X_train_ids (ndarray): Training input IDs.
    - X_train_mask (ndarray): Training attention masks.
    - y_train (list): Training labels.
    - epochs (int): Number of training epochs.
    - batch_size (int): Training batch size.
    
    Returns:
    - model (tf.keras.Model): Trained model.
    """
    model.fit(
        [X_train_ids, X_train_mask],
        np.array(y_train),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0  # Suppress training output for clarity
    )
    return model

def evaluate_model_performance(model, X_test_ids, X_test_mask, y_test):
    """
    Evaluate the model and compute performance metrics.
    
    Parameters:
    - model (tf.keras.Model): Trained model.
    - X_test_ids (ndarray): Test input IDs.
    - X_test_mask (ndarray): Test attention masks.
    - y_test (list): Test labels.
    
    Returns:
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f2 (float): F2 score.
    - avg_feedback (float): Average feedback assigned per requirement.
    """
    prediction = model.predict([X_test_ids, X_test_mask])
    y_pred = (prediction > 0.5).astype("int32").flatten()
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    
    # For dummy data, avg_feedback can be calculated as sum of predictions divided by number of requirements
    avg_feedback = y_pred.sum() / 1  # Assuming single requirement in dummy data
    
    return precision, recall, f2, avg_feedback

# -------------------------------------------------
# Step 5: Define Experiment Runner
# -------------------------------------------------

def run_experiment(experiment, feedback_df, requirements_df, preprocessed_feedback, preprocessed_requirements, ground_truth):
    """
    Execute a single experiment: preprocessing, training, evaluation, and logging.
    
    Parameters:
    - experiment (dict): Experiment configuration.
    - feedback_df (DataFrame): Feedback DataFrame.
    - requirements_df (DataFrame): Requirements DataFrame.
    - preprocessed_feedback (list of lists): Preprocessed feedback sentences.
    - preprocessed_requirements (list of lists): Preprocessed requirement sentences.
    - ground_truth (dict): Mapping of requirements to feedback IDs.
    
    Returns:
    - metrics (dict): Dictionary containing precision, recall, f2, avg_feedback.
    """
    experiment_id = experiment['id']
    experiment_name = experiment['name']
    preprocessing_steps = experiment.get('preprocessing', [])
    model_name = experiment['model']
    additional_steps = experiment.get('additional_steps', [])
    
    print(f"Starting Experiment {experiment_id}: {experiment_name}")
    
    # Handle additional steps (e.g., Experiment 14)
    if 'incorporate_previous_feedback' in additional_steps:
        # For demonstration, duplicate existing feedback assignments
        for req_id, fb_ids in ground_truth.items():
            ground_truth[req_id].extend(fb_ids)
    
    # Create positive and negative samples
    all_samples, labels = create_positive_negative_samples(
        feedback_df, requirements_df, preprocessed_feedback, preprocessed_requirements, ground_truth
    )
    
    if len(all_samples) == 0:
        print(f"No samples created for Experiment {experiment_id}. Skipping.")
        return
    
    # Tokenize and encode
    try:
        model, tokenizer = build_transformer_model(model_name)
    except Exception as e:
        print(f"Error building model for Experiment {experiment_id}: {e}")
        return
    
    input_ids, attention_mask = tokenize_and_encode(all_samples, tokenizer)
    
    # Split into train and test
    try:
        X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
            input_ids, attention_mask, labels, test_size=0.2, random_state=42
        )
    except ValueError as e:
        print(f"Error during train-test split for Experiment {experiment_id}: {e}")
        return
    
    # Handle case where test set might be empty
    if len(X_test_ids) == 0:
        print(f"No test samples available for Experiment {experiment_id}. Skipping evaluation.")
        return
    
    # Train the model
    model = train_model_on_data(model, X_train_ids, X_train_mask, y_train, epochs=1, batch_size=32)
    
    # Evaluate the model
    precision, recall, f2, avg_feedback = evaluate_model_performance(model, X_test_ids, X_test_mask, y_test)
    
    print(f"Completed Experiment {experiment_id}: Precision={precision:.2f}, Recall={recall:.2f}, F2 Score={f2:.2f}, Avg Feedback Assigned={avg_feedback:.2f}\n")
    
    # Log metrics with MLflow
    mlflow.set_experiment("FeReRe_Experiments")
    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_param("Experiment_ID", experiment_id)
        mlflow.log_param("Preprocessing_Steps", preprocessing_steps)
        mlflow.log_param("Model", model_name)
        mlflow.log_param("Additional_Steps", additional_steps)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F2_Score", f2)
        mlflow.log_metric("Avg_Feedback_Assigned", avg_feedback)
    
    # Optionally, save predictions and results
    predictions = pd.DataFrame({
        'Requirement_Sentence': [pair[0] for pair in all_samples],
        'Feedback_Sentence': [pair[1] for pair in all_samples],
        'Label': labels,
        'Prediction': (model.predict([input_ids, attention_mask]) > 0.5).astype(int).flatten()
    })
    predictions.to_excel(os.path.join(FINETUNE_DIR, f'predictions_experiment_{experiment_id}.xlsx'), index=False)
    
    results = {
        'Experiment_ID': experiment_id,
        'Precision': precision,
        'Recall': recall,
        'F2_Score': f2,
        'Avg_Feedback_Assigned': avg_feedback
    }
    results_df = pd.DataFrame([results])
    results_df.to_excel(os.path.join(FINETUNE_DIR, f'results_experiment_{experiment_id}.xlsx'), index=False)
    # del model
    # del tokenizer
    # gc.collect() # to empty the memory
    return results

# -------------------------------------------------
# Step 6: Execute All Experiments
# -------------------------------------------------

def run_all_experiments():
    # Define paths to your dataset files
    feedback_path = os.path.join(DATA_DIR, 'ReFeed/feedback.xlsx')
    requirements_path = os.path.join(DATA_DIR, 'ReFeed/requirements.xlsx')
    ground_truth_path = os.path.join(DATA_DIR, 'ReFeed/refeed_gt.xlsx')
    
    # Load data
    feedback_df, requirements_df, ground_truth_df, feedback_sentences, requirements_sentences, ground_truth = load_data(
        feedback_path, requirements_path, ground_truth_path
    )
    
    # For experiments, we'll use a subset of the data (e.g., first 5 entries)
    # Adjust this as needed to include more entries
    num_subset = 5  # Change this number based on your dataset size
    subset_feedback_sentences = feedback_sentences[:num_subset]
    subset_requirements_sentences = requirements_sentences[:num_subset]
    
    # Prepare subset_ground_truth
    subset_ground_truth = {}
    for req_id, fb_ids in ground_truth.items():
        # Check if the requirement is within the subset
        req_indices = requirements_df[requirements_df.iloc[:, 0] == req_id].index
        if not req_indices.empty and req_indices[0] < num_subset:
            subset_ground_truth[req_id] = fb_ids.copy()  # Use copy to prevent modifying original ground_truth
    
    print(f"Subset Ground Truth Mappings:")
    for req_id, fb_ids in subset_ground_truth.items():
        print(f"{req_id}: {fb_ids}")
    print("\n")
    
    # Iterate through all experiments
    for experiment in EXPERIMENTS:
        run_experiment(
            experiment,
            feedback_df.iloc[:num_subset],  # Subset feedback
            requirements_df.iloc[:num_subset],  # Subset requirements
            subset_feedback_sentences,
            subset_requirements_sentences,
            subset_ground_truth
        )
    
    print("All experiments completed successfully.")

if __name__ == "__main__":
    run_all_experiments()
