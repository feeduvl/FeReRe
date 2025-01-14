import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from datetime import datetime

from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize import sent_tokenize
import nltk

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import mixed_precision

# We import the FeReRe metrics functions to unify everything
from calculate_FeReRe_Metrics import load_excel, compute_metrics

def train_and_eval_kfold(
    feedback, 
    requirements, 
    ground_truth, 
    epochs=30, 
    n_splits=5,
    batch_size=64,
    subset_size=None,
    model_name='bert-base-uncased',
    exp_id=None
):
    """
    K-fold cross-validation approach for BERT-based text matching,
    using only FeReRe metrics from calculate_FeReRe_Metrics.py:
      - Mixed precision
      - MirroredStrategy for multi-GPU
      - EarlyStopping, TensorBoard per fold
      - Negative sample generation
      - Evaluate FeReRe metrics (Precision, Recall, F2, AvgAssigned)
      - Returns fold-wise FeReRe metrics
    """

    nltk.download('punkt')
    print(f"[INFO] Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # 1. Enable mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # 2. Load data
    feedback_df = pd.read_excel(feedback, header=None)
    requirements_df = pd.read_excel(requirements, header=None)
    ground_truth_df = pd.read_excel(ground_truth)

    # Optional subset sampling
    if subset_size is not None:
        if subset_size < len(feedback_df):
            feedback_df = feedback_df.sample(subset_size, random_state=42)
        if subset_size < len(requirements_df):
            requirements_df = requirements_df.sample(subset_size, random_state=42)

    feedback_df.reset_index(drop=True, inplace=True)
    requirements_df.reset_index(drop=True, inplace=True)

    # 3. Extract text columns
    feedback_texts = feedback_df.iloc[:, 1].tolist()
    requirements_texts = requirements_df.iloc[:, 1].tolist()

    # 4. Build ground truth dictionary { req_id : [feedback_ids] }
    ground_truth_dict = {}
    for col in ground_truth_df.columns:
        ground_truth_dict[col] = ground_truth_df[col].dropna().tolist()

    # 5. Sentence-split each requirement & feedback
    feedback_sentences = [sent_tokenize(text) for text in feedback_texts]
    requirements_sentences = [sent_tokenize(text) for text in requirements_texts]

    # 6. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # ---------------------------
    # Create positive samples
    # ---------------------------
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

    # ---------------------------
    # Create negative samples
    # ---------------------------
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

    # ---------------------------
    # Tokenize inputs
    # ---------------------------
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

    # Setup StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1

    # We'll store fold results here
    all_fold_metrics = []

    # For indexing back to requirement & feedback
    all_pairs = positive_pairs + negative_pairs

    # -----------------------------------------------
    # MAIN K-FOLD LOOP
    # -----------------------------------------------
    for train_indices, test_indices in skf.split(input_ids, np.array(labels)):
        print(f"\n[K-FOLD] Training fold {fold} for Experiment {exp_id}...\n")

        # 7. MirroredStrategy scope
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # 7a) Load BERT model
            bert_model = TFBertModel.from_pretrained(model_name)

            # 7b) Build model
            input_ids_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
            attention_mask_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

            outputs = bert_model(input_ids_layer, attention_mask=attention_mask_layer)
            pooled_output = outputs.pooler_output  # shape [batch, 768] for BERT-base
            # output = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)
            dropout = tf.keras.layers.Dropout(0.2)(pooled_output)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
            model = tf.keras.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)

            # 7c) Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy'] 
            )

        # 8. Split data
        X_train_ids, X_test_ids = input_ids[train_indices], input_ids[test_indices]
        X_train_mask, X_test_mask = attention_mask[train_indices], attention_mask[test_indices]
        y_train, y_test = np.array(labels)[train_indices], np.array(labels)[test_indices]
        
   

        # 9. TensorBoard / EarlyStopping
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(
            "./logs_kfold",
            f"Experiment_{exp_id}",
            f"fold_{fold}_{now_str}"
        )
        tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        )
        
        

        # 10. Train
        history = model.fit(
            [X_train_ids, X_train_mask],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, tb_callback],
            verbose=1)
        # # 10. Train (use the tf.data.Dataset objects)
        # history = model.fit(
        #     train_dataset,
        #     epochs=epochs,
        #     validation_data=val_dataset,
        #     callbacks=[early_stopping, tb_callback],
        #     verbose=1
        # )

       
        # 11. Inference for test pairs
        prediction = model.predict([X_test_ids, X_test_mask])
        y_pred = (prediction > 0.5).astype("int32")

        # 12. Build wide-format "classified_feedback"
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

        # 13. Save wide-format classified feedback
        max_len_fb = max(len(v) for v in predicted_results.values()) if predicted_results else 0
        for req_id in predicted_results:
            predicted_results[req_id] += [''] * (max_len_fb - len(predicted_results[req_id]))
        fold_classified_path = f'/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/classified_feedback_requirements_{exp_id}_fold_{fold}.xlsx'
        pd.DataFrame(predicted_results).to_excel(fold_classified_path, index=False)

        # 14. Build wide-format ground truth for just the test fold
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
        fold_gt_path = f'/nfs/home/vthakur_paech/FeReRe/data/finetuneBERT/kfold/test_ground_truth_{exp_id}_fold_{fold}.xlsx'
        pd.DataFrame(test_ground_truth).to_excel(fold_gt_path, index=False)

        # 15. Compute FeReRe metrics
        fold_classifier_df = load_excel(fold_classified_path)
        fold_gt_df = load_excel(fold_gt_path)
        precision_fere, recall_fere, f2_fere, avg_assign_fere = compute_metrics(fold_classifier_df, fold_gt_df)

        # 16. Print fold results (FeReRe only)
        print(f"[Fold {fold}] FeReRe => Precision={precision_fere:.4f}, "
              f"Recall={recall_fere:.4f}, F2={f2_fere:.4f}, "
              f"AvgAssign={avg_assign_fere:.2f}")

        # Store fold results
        fold_metrics = {
            'fold': fold,
            'precision': precision_fere,
            'recall': recall_fere,
            'f2': f2_fere,
            'avg_assign': avg_assign_fere
        }
        all_fold_metrics.append(fold_metrics)

        fold += 1

    # Summarize across folds
    if all_fold_metrics:
        avg_precision = np.mean([fm['precision'] for fm in all_fold_metrics])
        avg_recall = np.mean([fm['recall'] for fm in all_fold_metrics])
        avg_f2 = np.mean([fm['f2'] for fm in all_fold_metrics])
        avg_assign_overall = np.mean([fm['avg_assign'] for fm in all_fold_metrics])
        
        # print("\n[K-FOLD] ==============================")
        # print(f"Final Averages across {n_splits} folds for Exp {exp_id}:")
        # print(f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, "
        #       f"F2={avg_f2:.4f}, AvgAssign={avg_assign_overall:.2f}")
        # print("[K-FOLD] ==============================\n")
        
        # (D) ADDED: “Overall Performance” 
        
        print("\nOverall Performance:")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")
        print(f"Average F2 Score: {avg_f2:.2f}")
        print(f"Average Feedback per Requirement: {avg_assign_overall:.2f}")


    return all_fold_metrics
