from transformers import logging
logging.set_verbosity_error()
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from nltk.tokenize import sent_tokenize
import nltk
import random


def train_and_eval(feedback, requirements, ground_truth,
                   epochs=30,            # <-- (default epoch)
                   batch_size=64,       # <-- Increased batch size (was 32)
                   subset_size=None,
                   model_name='bert-base-uncased', exp_id=None):

    import os
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
    from datetime import datetime

    nltk.download('punkt')
    # # Ensure TensorFlow uses GPU
    # print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # 1. Enable mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # 2. MirroredStrategy for multi-GPU
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
    )
    
    with strategy.scope():
        print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
        
    
        # Load data
        feedback_df = pd.read_excel(feedback, header=None)
        requirements_df = pd.read_excel(requirements, header=None)
        ground_truth_df = pd.read_excel(ground_truth)

        if subset_size is not None:
            if subset_size < len(feedback_df):
                feedback_df = feedback_df.sample(subset_size, random_state=42)
            if subset_size < len(requirements_df):
                requirements_df = requirements_df.sample(subset_size, random_state=42)

        feedback_df.reset_index(drop=True, inplace=True)
        requirements_df.reset_index(drop=True, inplace=True)

        # Extract text
        feedback_texts = feedback_df.iloc[:, 1].tolist()
        requirements_texts = requirements_df.iloc[:, 1].tolist()

        # Prepare ground truth dictionary
        ground_truth_dict = {}
        for col in ground_truth_df.columns:
            ground_truth_dict[col] = ground_truth_df[col].dropna().tolist()

        feedback_sentences = [sent_tokenize(text) for text in feedback_texts]
        requirements_sentences = [sent_tokenize(text) for text in requirements_texts]

        # Load model and tokenizer dynamically
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformer_model = TFAutoModel.from_pretrained(model_name)

        # ----------------------
        # Create positive samples
        # ----------------------
        positive_samples = []
        positive_pairs = []
        for req_id, feedback_ids in ground_truth_dict.items():
            req_indices = requirements_df[requirements_df.iloc[:, 0] == req_id].index
            if not req_indices.empty:
                req_index = req_indices[0]
                req_sents = requirements_sentences[req_index]
                for fb_id in feedback_ids:
                    fb_indices = feedback_df[feedback_df.iloc[:, 0] == fb_id].index
                    if not fb_indices.empty:
                        fb_index = fb_indices[0]
                        fb_sents = feedback_sentences[fb_index]
                        for req_s in req_sents:
                            for fb_s in fb_sents:
                                positive_samples.append((req_s, fb_s))
                                positive_pairs.append((req_index, fb_index))

        # ----------------------
        # Create negative samples
        # ----------------------
        negative_samples = []
        negative_pairs = []
        while len(negative_samples) < len(positive_samples):
            req_index = random.choice(requirements_df.index)
            feedback_index = random.choice(feedback_df.index)
            if (req_index, feedback_index) not in positive_pairs:
                req_sents = requirements_sentences[req_index]
                fb_sents = feedback_sentences[feedback_index]
                for req_s in req_sents:
                    for fb_s in fb_sents:
                        negative_samples.append((req_s, fb_s))
                        negative_pairs.append((req_index, feedback_index))
                        if len(negative_samples) >= len(positive_samples):
                            break
                    if len(negative_samples) >= len(positive_samples):
                        break

        # Combine samples
        all_samples = positive_samples + negative_samples
        labels = [1] * len(positive_samples) + [0] * len(negative_samples)

        # ----------------------
        # Tokenize inputs
        # ----------------------
        max_length = 128
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

        # ----------------------
        # Train / test split
        # ----------------------
        X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
            input_ids,
            attention_mask,
            labels,
            test_size=0.2,
            random_state=42
        )

        # ----------------------
        # Build model
        # ----------------------
        input_ids_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
        attention_mask_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

        outputs = transformer_model(input_ids_layer, attention_mask=attention_mask_layer)
        # For BERT-based models: outputs.last_hidden_state is [batch, seq_len, hidden]
        # We'll just take the [CLS] token (index 0) as the pooled representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)

        model = tf.keras.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # --------------------------------------
    # EarlyStopping callback + TensorBoard)
    # --------------------------------------
    #This will stop training if val_loss does not improve for 'patience' epochs.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,          # Stop after 4 consecutive epochs of no improvement
        restore_best_weights=True
    )
     # Root directory for this experiment
    root_log_dir = os.path.join(
        "./logs",
        f"Experiment_{exp_id}",
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    # Subdirectories for train and validation logs
    train_log_dir = os.path.join(root_log_dir, "train")
    val_log_dir   = os.path.join(root_log_dir, "validation")


    
    tensorboard_callback = TensorBoard(
        log_dir=root_log_dir,
        histogram_freq=0  # Set to 1 if you want to record histograms of weights (can be more resource-intensive)
    )
    # ----------------------
    # Train model
    # ----------------------
    model.fit(
        [X_train_ids, X_train_mask],
        np.array(y_train),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, tensorboard_callback]
    )

    # ----------------------
    # Predict on test set
    # ----------------------
    prediction = model.predict([X_test_ids, X_test_mask])
    y_pred = (prediction > 0.5).astype("int32")

    # Compute metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

    # Identify test pairs
    all_pairs = positive_pairs + negative_pairs
    _, test_indices, _, _ = train_test_split(
        np.arange(len(all_pairs)), labels, test_size=0.2, random_state=42
    )

    # Build dictionary of predicted results
    predicted_results = {req_id: [] for req_id in requirements_df.iloc[:, 0]}
    req_feedback_counts = {req_id: 0 for req_id in requirements_df.iloc[:, 0]}

    for idx, y_pred_value in zip(test_indices, y_pred):
        req_index, fb_index = all_pairs[idx]
        if y_pred_value == 1:
            req_id = requirements_df.iloc[req_index, 0]
            fb_id = feedback_df.iloc[fb_index, 0]
            if fb_id not in predicted_results[req_id]:
                predicted_results[req_id].append(fb_id)
                req_feedback_counts[req_id] += 1

    avg_feedback_per_requirement = np.mean(list(req_feedback_counts.values()))

    print(f'Precision: {precision}, Recall: {recall}, F2 Score: {f2}, Avg Assign: {avg_feedback_per_requirement}')

    # ----------------------
    # Save metrics
    # ----------------------
    results = {
        'Precision': [precision],
        'Recall': [recall],
        'F2 Score': [f2],
        "Avg Assigned": [avg_feedback_per_requirement]
    }
    pd.DataFrame(results).to_excel(
        '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/bert_results.xlsx',
        index=False
    )

    # Build the final DataFrame for assigned feedback
    max_len_feedback = max(len(v) for v in predicted_results.values()) if predicted_results else 0
    for req_id in predicted_results:
        predicted_results[req_id] += [''] * (max_len_feedback - len(predicted_results[req_id]))
    pd.DataFrame(predicted_results).to_excel(
        '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/classified_feedback_requirements.xlsx',
        index=False
    )

    # Build the final test ground-truth DataFrame
    test_pairs = [all_pairs[i] for i in test_indices]
    test_ground_truth = {req_id: [] for req_id in requirements_df.iloc[:, 0]}
    for req_index, fb_index in test_pairs:
        req_id = requirements_df.iloc[req_index, 0]
        fb_id = feedback_df.iloc[fb_index, 0]
        if fb_id in ground_truth_dict.get(req_id, []) and fb_id not in test_ground_truth[req_id]:
            test_ground_truth[req_id].append(fb_id)

    max_len_gt = max(len(v) for v in test_ground_truth.values()) if test_ground_truth else 0
    for req_id in test_ground_truth:
        test_ground_truth[req_id] += [''] * (max_len_gt - len(test_ground_truth[req_id]))

    pd.DataFrame(test_ground_truth).to_excel(
        '/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/test_ground_truth.xlsx',
        index=False
    )
