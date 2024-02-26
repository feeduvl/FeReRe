from jiraExtractor import extractIssuesFromJira
from bertEmbeddingHandler import process_excel
import CosSimCalculator
from thresholdFilter import treshold_filter
from Evaluator import evaluate
import os
import glob

def runEvalBertNoPrefix():
    print("Extracting Jira Issues")
    extractIssuesFromJira("data/", True)
    print("Getting Issue Embeddings")
    process_excel('data/jira_issues_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    print("Getting Feedback Embeddings")
    process_excel('data/feedback.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    print("Calculating Cosine Similarity")
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    print("Creating Threshold-Filtered files")
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    print("Evaluating")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_NoPrefix")
    print("Cleaning")
    # Specify the directory path
    directory_path = 'data/bert/thresholds'
    # Use glob to match all files within the directory
    files = glob.glob(os.path.join(directory_path, '*'))
    # Loop over the list of filepaths & remove each file
    for file_path in files:
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            os.remove(file_path)
    print("All threshold files have been removed from the directory.")

def runEvalBertNoPrefixNoNames():
    print("Extracting Jira Issues")
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    print("Getting Issue Embeddings")
    process_excel('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    print("Getting Feedback Embeddings")
    process_excel('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    print("Calculating Cosine Similarity")
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    print("Creating Threshold-Filtered files")
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    print("Evaluating")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_NoPrefix_NoNames")
    print("Cleaning")
    # Specify the directory path
    directory_path = 'data/bert/thresholds'
    # Use glob to match all files within the directory
    files = glob.glob(os.path.join(directory_path, '*'))
    # Loop over the list of filepaths & remove each file
    for file_path in files:
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            os.remove(file_path)
    print("All threshold files have been removed from the directory.")

#runEvalBertNoPrefix()
#runEvalBertNoPrefixNoNames()



