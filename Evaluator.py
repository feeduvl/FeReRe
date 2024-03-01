import pandas as pd
import numpy as np
import os
import glob

class PrecisionRecallEvaluator:
    def __init__(self, ground_truth_path, excel_files_path, output_path, chosen_feedback, removeNoRel):
        self.chosen_feedback = chosen_feedback
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.excel_files_path = excel_files_path
        self.output_path = output_path
        self.removeNoRel = removeNoRel

    def load_ground_truth(self, path):
        df = pd.read_excel(path, header=None)
        ground_truth = {}
        for column in df:
            issue_id = df.iloc[0, column]
            feedback_ids = set(df.iloc[1:, column].dropna().astype(str))
            ground_truth[str(issue_id)] = feedback_ids  # Ensure issue IDs are strings for consistent comparison
        if self.chosen_feedback != None:
            for issue_key in self.chosen_feedback:
                ground_truth[issue_key].remove(self.chosen_feedback[issue_key])
        return ground_truth

    def evaluate_files(self):
        all_results = pd.DataFrame()

        for filename in os.listdir(self.excel_files_path):
            if filename.endswith('.xlsx'):
                threshold = filename.replace('.xlsx', '').split('_')[-1]  # Extract threshold from filename
                file_path = os.path.join(self.excel_files_path, filename)
                df = pd.read_excel(file_path, index_col=0)
                results, avg_predicted_assignments = self.evaluate_file(df)

                # Calculate average metrics
                avg_precision = results['Precision'].mean()
                avg_recall = results['Recall'].mean()
                avg_f1_score = results['F1-Score'].mean()

                # Append average metrics to the results DataFrame
                results.loc['Average'] = [avg_precision, avg_recall, avg_f1_score]

                # Append the average number of predicted assignments
                results.loc['Avg. Assign'] = [avg_predicted_assignments, np.nan, np.nan]

                # Add an empty column (without a name) for spacing between thresholds
                if threshold!="0.00":
                    all_results[f'{threshold} '] = np.nan  # Using a space as the column name
                # Add the results to the all_results DataFrame
                for metric in results.columns:
                    all_results[f'{threshold} {metric}'] = results[metric]


        # Save all results to a single Excel file
        all_results.to_excel(self.output_path+'.xlsx')

    def evaluate_file(self, df):
        precision_list, recall_list, f1_list = [], [], []
        total_predicted_assignments = 0

        for issue_id in df.columns:
            true_feedback_ids = self.ground_truth.get(issue_id, set())
            predicted_feedback_ids = set(df[df[issue_id].notna()].index.astype(str))
            if self.removeNoRel == True and len(true_feedback_ids)<1:
                precision_list.append(np.NaN)
                recall_list.append(np.NaN)
                f1_list.append(np.NaN)
                continue
            if self.chosen_feedback != None and issue_id in self.chosen_feedback:
                predicted_feedback_ids.discard(self.chosen_feedback[issue_id])
            total_predicted_assignments += len(predicted_feedback_ids)

            true_positives = len(predicted_feedback_ids & true_feedback_ids)
            false_positives = len(predicted_feedback_ids - true_feedback_ids)
            false_negatives = len(true_feedback_ids - predicted_feedback_ids)

            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)
        avg_predicted_assignments = total_predicted_assignments / len(df.columns) if df.columns.size > 0 else 0

        # Include the issue IDs as the index for the DataFrame
        results_df = pd.DataFrame({
            'Precision': precision_list,
            'Recall': recall_list,
            'F1-Score': f1_list
        }, index=df.columns)

        # Add averages at the end
        results_df.loc['Average'] = [np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)]

        return results_df, avg_predicted_assignments

def clean_threshold_files(threshold_files):
    print("Cleaning")
    directory_path = threshold_files
    files = glob.glob(os.path.join(directory_path, '*'))
    for file_path in files:
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("All threshold files have been removed from the directory.")

def evaluate(gold_standard, threshold_files,output_path, chosen_feedback=None, removeNoRel=False, cleanup=True):
    print("Evaluating")
    calculator = PrecisionRecallEvaluator(gold_standard, threshold_files,output_path, chosen_feedback, removeNoRel)
    calculator.evaluate_files()
    if cleanup:
        clean_threshold_files(threshold_files)
