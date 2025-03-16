# FeReRe K-Fold Experiments

This repository contains code to evaluate the FeReRe (Feedback Requirements Relationship) metrics through a series of K-Fold experiments. It leverages transformer-based models for classifying feedback and requirements, applies several text preprocessing techniques, and logs experiment parameters and metrics using MLflow.

## Dataset

Contains feedback, requirements and ground truth for available datasets. This includes:
  
  - **komoot**
    - App reviews crawled from the google play store about the hiking app Komoot
    - Requirements recreated using the TORE Framework
  - **ReFeed**
    - Requirements & Feedback from the dataset available through the following publication
      - "Automating user-feedback driven requirements prioritization": https://www.sciencedirect.com/science/article/pii/S0950584921001014
  - **smartage**
      - Feedback and requirements for the SmartVernetzt and SmartFeedback apps
      - Feedback collected through the SmartFeedback app as part of the SmartAge-Project (feedback not available here due to data privacy laws)
      - Requirements created by the apps developers using the TORE Framework

## Files

### `calculate_FeReRe_Metrics.py`
Contains functions to:
- Load Excel files containing classification and ground-truth data.
- Compute FeReRe metrics such as precision, recall, F2 score, and average feedback per requirement.
- Log parameters and metrics to MLflow.

### `kfold_combined.py`
Implements the main experiment workflow:
- Loads and combines feedback, requirements, and ground-truth data from multiple Excel files.
- Applies text preprocessing (stopword removal, synonym-based augmentation, sentence splitting).
- Creates positive and negative samples for feedback–requirement pairs.
- Runs K-Fold cross-validation using transformer models (e.g., BERT, RoBERTa, DistilBERT).
- Trains a classifier and evaluates its performance per fold.
- Saves classification results and computes FeReRe metrics for each fold.
- Logs experiment details and metrics with MLflow.

### `run_kfoldexp.py`
Acts as the entry point for running multiple K-Fold experiments with various configurations:
- Defines a list of 15 experiment configurations with varying preprocessing and model settings.
- For experiment 14, it incorporates previously assigned feedback by appending feedback text to the requirements text before evaluation.
- Calls the `train_eval_kfold_combined` function from `kfold_combined.py` to run each experiment.
- Saves output files (Excel files for classified feedback and ground truth) and logs metrics via MLflow.

## Requirements

**Python:** 3.10

**Packages:**
- `pandas`
- `scikit-learn`
- `mlflow`
- `transformers`
- `tensorflow`
- `nltk`

Before running the code, install the dependencies. For example, you can use pip:

```bash
pip install pandas scikit-learn mlflow transformers tensorflow nltk
```
Also, ensure you have the necessary NLTK data packages:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
## Run Experiments


```bash
python run_kfoldexp.py
```
The script will process multiple experiment configurations. For each experiment (identified by an exp_id), it does the following steps:

* Load and preprocess the data.
* Run a Stratified K-Fold cross-validation.
* Save classification results and ground-truth files for each fold in the results directory (./new_kfold_results).
* Log detailed parameters and metrics with MLflow.

## Experiment Configurations

The experiments vary in the following ways:

* **Preprocessing Options:**

    Options include stopword removal, synonym-based data augmentation, and      sentence splitting.

* **Model Selection:**

    The experiments support different transformer-based models (e.g.,   bert-base-uncased, roberta-base, bert-large-uncased, distilbert-base-uncased).
    
* **Special Experiment (Exp 14):**

    This configuration incorporates previously assigned feedback into the requirements by appending feedback text to the requirement text before classification.
    
## Output Files

For each experiment and fold, the following files are generated:

* Classified Feedback (Wide-format):

    **classified_feedback_requirements_exp_<exp_id>_fold_<fold>.xlsx**
    
* Test Ground Truth (Wide-format):

    **test_ground_truth_exp_<exp_id>_fold_<fold>.xlsx**
    
These files are stored in the results directory (./new_kfold_results).

## MLflow Logging

MLflow is used to record:

* Experiment parameters (e.g., model type, epochs, batch size, preprocessing settings).

* Performance metrics (precision, recall, F2 score, average assigned feedback) for each fold and overall.

To view the logs, launch the MLflow UI:

```bash
mlflow ui
```

Then open your browser and navigate to **http://localhost:5000**.

## Additional Notes

* **GPU Memory Growth:**

    The code includes settings to enable GPU memory growth for TensorFlow, which can help when using large transformer models.
    
* **Data Format:**

    Ensure that your Excel files follow the expected format:
    * The first row typically contains IDs.
    * Subsequent rows contain text or feedback entries.
    
* **Customization:**

    Adjust experiment configurations, preprocessing methods, or model parameters as needed.
    
    



