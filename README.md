# FeReRe 
This repository contains the files necessary to train and evaluate the Feedback Requirements Relation using the FeReRe approach.

# Folder Structure
## data
Contains feedback, requirements and ground truth for available datasets. This includes:
  
  - komoot
    - App reviews crawled from the google play store about the hiking app Komoot
    - Requirements recreated using the TORE Framework
  - ReFeed
    - Requirements & Feedback from the dataset available through the following publication
      - "Automating user-feedback driven requirements prioritization": https://www.sciencedirect.com/science/article/pii/S0950584921001014
  - smartage
      - Feedback and requirements for the SmartVernetzt and SmartFeedback apps
      - Feedback collected through the SmartFeedback app as part of the SmartAge-Project (feedback not available here due to data privacy laws)
      - Requirements created by the apps developers using the TORE Framework

The "finetuneBERT" folder will contain the classifiers results as excel files

## experiments
Contains all Python files for training and evaluation of the bert classifier.

  - finetuneBERT.py
    - Training on a single dataset for the specified amount of epochs
  - finetuneBERT_kfold.py
    - K-Fold training on a single dataset
  - finetuneBERT_multipleDatasets.py
    - Train on multiple datasets consecutively and evaluate consecutively 
  - finetuneBERT_traintestdifferentdata.py
    - Train on one set of data and evaluate classifier on another
  - finetuneBERT_combinedata.py
    - Train and evaluate on a combination of all datasets simultaneously
   
Please note that the metrics output by the above files are not the results of FeReRe but merely the metrics for the similarity classification of individual sentences. To evaluate FeReRe use *calculate_FeReRe_Metrics.py*.

This file compares the Feedback Requirements Relation of the classifier to the Ground Truth and calculates the true Precision, Recall and F2 by comparing the Excel-files created after training. Results are printed and can optionally be tracked with an MLFlow setup using the following command in the Repository root directory:
```
mlflow ui --backend-store-uri file:mlruns
```
## old_experiments
Contains failed experiments for Feedback Requirements Relation that showed much poorer results than the finetuned BERT approach. These experiments are no longer maintained.

