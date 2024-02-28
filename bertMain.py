from jiraExtractor import extractIssuesFromJira
from bertEmbeddingHandler import create_embeddings, create_combined_embeddings, create_average_embeddings
import CosSimCalculator
from thresholdFilter import treshold_filter
from Evaluator import evaluate


def runEvalBertNoPrefix():
    extractIssuesFromJira("data/", True)
    create_embeddings('data/jira_issues_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix")

def runEvalBertNoPrefixNoNames():
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix_NoNames")

def runEvalBertNoPrefixNoNamesFewer1Assign():
    #extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    #create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    #create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix_NoNames_Fewer1Assign",removeNoRel=True)

def runEvalBertNoPrefixNoNamesAvgCosSim():
    # Treats one feedback per issue from Ground Truth as already assigned and calculates an average similarity score
    # of CosSim(Issue,NewFeedback) and CosSim(AssignedFeedback,Newfeedback)
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    # Got to run at least from here to get chosen_feedback:
    chosen_feedback = CosSimCalculator.calc_cos_sim_incl_one_feedback_avgcossim('data/bert/issues_bert_embeddings.xlsx',
                                                                                'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_AvgCosSim_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_AvgCosSim", chosen_feedback)

def runEvalBertNoPrefixNoNamesAvgCosSimFewer1Assign():
    # Treats one feedback per issue from Ground Truth as already assigned and calculates an average similarity score
    # of CosSim(Issue,NewFeedback) and CosSim(AssignedFeedback,Newfeedback)
    extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    # to run at least from here to get chosen_feedback:
    chosen_feedback = CosSimCalculator.calc_cos_sim_incl_one_feedback_avgcossim('data/bert/issues_bert_embeddings.xlsx',
                                                                                'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("data/bert/bert_AvgCosSim_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_AvgCosSimFewer1Assign",
             chosen_feedback, True)

def runEvalBertNoPrefixNoNamesCombinedEmbeddings():
    # Combine Text of Issue and 1 Feedback and calculate that combined embedding
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_combined_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_CombinedEmbeddings', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesCombinedEmbeddingsFewer1Assign():
    # Combine Text of Issue and 1 Feedback and calculate that combined embedding
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_combined_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_CombinedEmbeddingsFewer1Assign', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesHighestSimScore():

    # Combine Text of Issue and 1 Feedback and calculate that combined embedding
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    chosen_feedback=CosSimCalculator.calc_cos_sim_incl_one_feedback_highestscore('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter( 'data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_HighestSimScore', chosen_feedback)

def runEvalBertNoPrefixNoNamesAvgEmbeddings():
    # Calculate average of issue and feedback embedding and calculate cos sim for new feedback with average
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_average_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_AverageEmbeddings', chosen_feedback)

def runEvalBertNoPrefixNoNamesAvgEmbeddingsFewer1Assign():
    # Calculate average of issue and feedback embedding and calculate cos sim for new feedback with average
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_average_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_AverageEmbeddingsFewer1Assign', chosen_feedback, True)

# runEvalBertNoPrefix()
# runEvalBertNoPrefixNoNames()
# Results: P 0.29, R 0.43, F1 0.29 Avg: 13 at 0.85
# runEvalBertNoPrefixNoNamesFewer1Assign()
# Results: P 0.30, R 0.45, F1 0.31 Avg: 13 at 0.85
# runEvalBertNoPrefixNoNamesAvgCosSim()
# Results: P 0.1, R 0.53, F1 0.14 Avg: 39
# runEvalBertNoPrefixNoNamesAvgCosSimFewer1Assign()
# Results: P 0.14 R 0.67, F1 0.19 Avg: 34
# runEvalBertNoPrefixNoNamesCombinedEmbeddings()
# Results: P 0.16 R 0.42 F1 0.17 Avg: 22 at 0.89
# runEvalBertNoPrefixNoNamesCombinedEmbeddingsFewer1Assign()
# Results: P 0.18 R 0.47 F1 0.20 Avg: 18 at 0.89
# runEvalBertNoPrefixNoNamesHighestSimScore()
# Results: P 0.10 R 0.35 F1 0.13 Avg: 34 at 0.88
# runEvalBertNoPrefixNoNamesAvgEmbeddings()
# Results: P 0.15 R 0.41 F1 0.19 Avg: 20 at 0.9
#runEvalBertNoPrefixNoNamesAvgEmbeddingsFewer1Assign()
# Results: P 0.19 R 0.47 F1 0.23 Avg: 17 at 0.9
#Todo AvgEmbeddings with more than 1 Feedback