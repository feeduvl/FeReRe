import pandas as pd
import numpy as np



# Function to filter DataFrame based on threshold
def filter_df_by_threshold(df, threshold):
    # Apply the threshold, setting values below it to NaN
    filtered_df = df.applymap(lambda x: x if x >= threshold else np.nan)
    return filtered_df

def treshold_filter(cos_sim_file,output_file_path):
    # Load the initial Excel file containing the similarity scores
    initial_file = cos_sim_file  # Adjust the file name as necessary
    results_df = pd.read_excel(initial_file, index_col=0)  # Assuming the first column contains the index

    # Iterate over the range from 0 to 1 in steps of 0.01
    for threshold in np.arange(0, 1.01, 0.01):
        # Filter the DataFrame by the current threshold
        filtered_df = filter_df_by_threshold(results_df, threshold)

        # Define the file name based on the current threshold
        file_name = output_file_path+f'/bert_similarity_scores_threshold_{threshold:.2f}.xlsx'

        # Save the filtered DataFrame to an Excel file
        filtered_df.to_excel(file_name)
