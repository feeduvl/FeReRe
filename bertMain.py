from jiraExtractor import extractIssuesFromJira
from bertEmbeddingHandler import process_excel
import CosSimCalculator
from thresholdFilter import treshold_filter
from Evaluator import evaluate
import os
import glob

def clean_threshold_files():
    print("Cleaning")
    directory_path = 'data/bert/thresholds'
    files = glob.glob(os.path.join(directory_path, '*'))
    for file_path in files:
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("All threshold files have been removed from the directory.")
def runEvalBertNoPrefix():
    extractIssuesFromJira("data/", True)
    process_excel('data/jira_issues_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    process_excel('data/feedback.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_NoPrefix")
    clean_threshold_files()

def runEvalBertNoPrefixNoNames():
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    process_excel('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    process_excel('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_NoPrefix_NoNames")
    clean_threshold_files()

def runEvalBertNoPrefixNoNamesAvgCosSim():
    #Treats one feedback per issue from Ground Truth as already assigned and calculates an average similarity score of CosSim(Issue,NewFeedback) and CosSim(AssignedFeedback,Newfeedback)
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    process_excel('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    process_excel('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    #Got to run at least from here to get chosen_feedback:
    chosen_feedback=CosSimCalculator.calc_cos_sim_incl_one_feedback_avgcossim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_AvgCosSim_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_AvgCosSim",chosen_feedback)
    clean_threshold_files()

def runEvalBertNoPrefixNoNamesAvgCosSimFewer1Assign():
    #Treats one feedback per issue from Ground Truth as already assigned and calculates an average similarity score of CosSim(Issue,NewFeedback) and CosSim(AssignedFeedback,Newfeedback)
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    process_excel('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    process_excel('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    # to run at least from here to get chosen_feedback:
    chosen_feedback=CosSimCalculator.calc_cos_sim_incl_one_feedback_avgcossim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_AvgCosSim_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_AvgCosSimFewer1Assign",chosen_feedback,True)
    clean_threshold_files()

def runEvalBertNoPrefixNoNamesCombinedEmbeddings():
    #Combine Text of Issue and 1 Feedback and calculate that combined embedding
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    process_excel('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    process_excel('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx',"data/bert/thresholds","data/bert/Eval_Bert_NoPrefix_NoNames")
    clean_threshold_files()

#runEvalBertNoPrefix()
#runEvalBertNoPrefixNoNames()
#Results: P 0.29, R 0.43, F1 0.29 Avg: 13
#runEvalBertNoPrefixNoNamesAvgCosSim()
#Results: P 0.1, R 0.53, F1 0.14 Avg: 39
#runEvalBertNoPrefixNoNamesAvgCosSimFewer1Assign()
#Results: P 0.14 R 0.67, F1 0.19 Avg: 34
runEvalBertNoPrefixNoNamesCombinedEmbeddings()

#Take highest Sim_score of all