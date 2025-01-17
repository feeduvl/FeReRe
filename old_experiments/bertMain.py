from old_experiments.jiraExtractor import extractIssuesFromJira
from bertEmbeddingHandler import create_embeddings, create_combined_embeddings, create_average_embeddings, \
    create_TFIDFweightedaverage_embeddings, create_TFIDF_embeddings, create_concatenated_embeddings
import CosSimCalculator
from old_experiments.thresholdFilter import treshold_filter
from old_experiments.Evaluator import evaluate


def runEvalBertNoPrefix():
    # Jira Issue Prefixes are removed from requirements
    #extractIssuesFromJira("data/", True)
    #create_embeddings('data/jira_issues_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    #create_embeddings('data/feedback.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    #CosSimCalculator.calc_cos_sim('data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    #treshold_filter("data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    #evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix")

def runEvalBertNoPrefixNoNames():
    # Jira Issue Prefixes and Software specific Names such as "Komoot" are removed from requirements
    extractIssuesFromJira("../data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', '../data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback_nonames.xlsx', '../data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('../data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("../data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix_NoNames")

def runEvalBertNoPrefixNoNamesFewer1Assign():
    # Fewer1Assign means that feedback that is not related to any requirement in the ground truth is discarded before classification
    #extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    #create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    #create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('../data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("../data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix_NoNames_Fewer1Assign",removeNoRel=True)

def runEvalBertNoPrefixNoNamesTFIDFFewer1Assign():
    # Word Embeddings are weighted by the TFIDF scores of the individual words in the feedback and requirements
    #extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_TFIDF_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx','data/feedback_nonames.xlsx',
                            '../data/bert/issues_bert_embeddings.xlsx')
    create_TFIDF_embeddings('data/feedback_nonames.xlsx','data/jira_issues_namesfiltered_noprefix.xlsx',
                            '../data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('../data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("../data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_NoPrefix_NoNames_TFIDF_Fewer1Assign",removeNoRel=True)

def runEvalBertNoPrefixNoNamesAvgCosSim():
    # Treats one feedback per issue from Ground Truth as already assigned and calculates an average similarity score
    # of CosSim(Issue,NewFeedback) and CosSim(AssignedFeedback,Newfeedback)
    extractIssuesFromJira("../data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', '../data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback_nonames.xlsx', '../data/bert/feedback_bert_embeddings.xlsx')
    # Got to run at least from here to get chosen_feedback:
    chosen_feedback = CosSimCalculator.calc_cos_sim_incl_one_feedback_avgcossim(
        '../data/bert/issues_bert_embeddings.xlsx',
                                                                                'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("../data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_AvgCosSim", chosen_feedback)

def runEvalBertNoPrefixNoNamesAvgCosSimFewer1Assign():
    # Treats one feedback per issue from Ground Truth as already assigned and calculates an average similarity score
    # of CosSim(Issue,NewFeedback) and CosSim(AssignedFeedback,Newfeedback)
    extractIssuesFromJira("../data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', '../data/bert/issues_bert_embeddings.xlsx')
    create_embeddings('data/feedback_nonames.xlsx', '../data/bert/feedback_bert_embeddings.xlsx')
    # to run at least from here to get chosen_feedback:
    chosen_feedback = CosSimCalculator.calc_cos_sim_incl_one_feedback_avgcossim(
        '../data/bert/issues_bert_embeddings.xlsx',
                                                                                'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter("../data/bert/bert_similarity_scores.xlsx", "data/bert/thresholds")
    evaluate('data/Ground_Truth.xlsx', "data/bert/thresholds", "data/bert/Eval_Bert_AvgCosSimFewer1Assign",
             chosen_feedback, True)

def runEvalBertNoPrefixNoNamesCombinedEmbeddings():
    # Combine Text of Issue and 1 Feedback and calculate that combined embedding
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_combined_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_CombinedEmbeddings', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesCombinedEmbeddingsFewer1Assign():
    # Combine Text of Issue and 1 Feedback and calculate that combined embedding
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_combined_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_CombinedEmbeddingsFewer1Assign', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesHighestSimScore():
    # Treat one feedback as already assigned. Calculate sim score of new feedback and issue and new feedback and already assigned feedback and take the higher sim score of the two
    #extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    #create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/bert/issues_bert_embeddings.xlsx')
    #create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    chosen_feedback= CosSimCalculator.calc_cos_sim_incl_one_feedback_highestscore(
        '../data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_HighestSimScore', chosen_feedback)

def runEvalBertNoPrefixNoNamesHighestSimScoreFewer1Assign():
    # Treat one feedback as already assigned. Calculate sim score of new feedback and issue and new feedback and already assigned feedback and take the higher sim score of the two
    #extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    create_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', '../data/bert/issues_bert_embeddings.xlsx')
    #create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    chosen_feedback= CosSimCalculator.calc_cos_sim_incl_one_feedback_highestscore(
        '../data/bert/issues_bert_embeddings.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_HighestSimScoreFewer1Assign', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesAvgEmbeddings():
    # Calculate average of issue and feedback embedding and calculate cos sim for new feedback with average
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_average_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_AverageEmbeddings', chosen_feedback)

def runEvalBertNoPrefixNoNamesAvgEmbeddingsFewer1Assign():
    # Calculate average of issue and feedback embedding and calculate cos sim for new feedback with average
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_average_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    #create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_AverageEmbeddingsFewer1Assign', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesTFIDFAvgEmbeddingsFewer1Assign():
    # Calculate average of issue and feedback embedding and calculate cos sim for new feedback with average including TFIDF weights
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_TFIDFweightedaverage_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    create_TFIDF_embeddings('data/feedback_nonames.xlsx','data/jira_issues_namesfiltered_noprefix.xlsx',
                            '../data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_TFIDFAverageEmbeddingsFewer1Assign', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesAvgEmbeddingsMultipleFeedbackFewer1Assign():
    # Calculate average of issue and multiple feedback embeddings and calculate cos sim for new feedback with average
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_average_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx', 2)
    # create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_AverageEmbeddingsMultiFeedbackFewer1Assign', chosen_feedback, True)

def runEvalBertNoPrefixNoNamesConcatenatedEmbeddingsFewer1Assign():
    # Concatenate Embeddings of issue and already assigned feedback, reduce and calculate cos sim with new feedback
    # extractIssuesFromJira("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
    chosen_feedback = create_concatenated_embeddings('data/jira_issues_namesfiltered_noprefix.xlsx', 'data/feedback_nonames.xlsx', 'data/bert/issues_bert_combined_embeddings.xlsx', 'data/Ground_Truth.xlsx')
    #create_embeddings('data/feedback_nonames.xlsx', 'data/bert/feedback_bert_embeddings.xlsx')
    CosSimCalculator.calc_cos_sim('data/bert/issues_bert_combined_embeddings.xlsx',
                                  '../data/bert/feedback_bert_embeddings.xlsx', chosen_feedback)
    treshold_filter('../data/bert/bert_similarity_scores.xlsx', 'data/bert/thresholds')
    evaluate('data/Ground_Truth.xlsx', 'data/bert/thresholds', 'data/bert/Eval_Bert_ConcatenatedEmbeddingsFewer1Assign', chosen_feedback, True)

runEvalBertNoPrefix()


#Old Results with old data
# runEvalBertNoPrefixNoNames()
# Results: P 0.29 R 0.43 F1 0.29 Avg: 13 at 0.85
# Results: P 0.06 R 0.85 F1 0.09 Avg: 164 at 0.42 Only Issue Summary
# runEvalBertNoPrefixNoNamesFewer1Assign()
# Results: P 0.30 R 0.45 F1 0.31 Avg: 13 at 0.85
# runEvalBertNoPrefixNoNamesTFIDFFewer1Assign()
# Results: P 0.28 R 0.18 F1 0.17 AVG: 04 at 0.85
# Results: P 0.24 R 0.49 F1 0.26 Avg: 20 at 0.82
# runEvalBertNoPrefixNoNamesAvgCosSim()
# Results: P 0.1, R 0.53, F1 0.14 Avg: 39
# runEvalBertNoPrefixNoNamesAvgCosSimFewer1Assign()
# Results: P 0.14 R 0.67, F1 0.19 Avg: 34
# runEvalBertNoPrefixNoNamesCombinedEmbeddings()
# runEvalBertNoPrefixNoNamesCombinedEmbeddingsFewer1Assign()
# Results: P 0.09 R 0.91 F1 0.13 Avg 79 at 0.85
# Results: P 0.23 R 0.51 F1 0.24 Avg 16 at 0.89
# runEvalBertNoPrefixNoNamesHighestSimScore()
# runEvalBertNoPrefixNoNamesHighestSimScoreFewer1Assign()
# Results: P 0.07 R 0.94 F1 0.11 Avg: 96 at 0.85
# Results: P 0.11 R 0.54 F1 0.15 Avg: 41 at 0.88
# runEvalBertNoPrefixNoNamesAvgEmbeddings()
# runEvalBertNoPrefixNoNamesAvgEmbeddingsFewer1Assign()
# Results: P 0.06 R 0.98 F1 0.11 Avg: 99 at 0.85
# Results: P 0.26 R 0.34 F1 0.25 Avg: 09 at 0.91
# runEvalBertNoPrefixNoNamesTFIDFAvgEmbeddingsFewer1Assign()
# Results: P 0.10 R 0.80 F1 0.15 Avg: 60 at 0.85
# Results: P 0.19 R 0.49 F1 0.23 Avg: 18 at 0.88
# runEvalBertNoPrefixNoNamesAvgEmbeddingsMultipleFeedbackFewer1Assign()
# Results: P 0.06 R 0.97 F1 0.10 Avg: 101 at 0.85
# Results: P 0.26 R 0.48 F1 0.29 Avg: 11 at 0.91
# runEvalBertNoPrefixNoNamesConcatenatedEmbeddingsFewer1Assign()
# Results: P 0.06 R 0.99 F1 0.11 Avg: 100 at 0.85
# Results: P 0.20 R 0.40 F1 0.22 Avg: 15 at 0.90