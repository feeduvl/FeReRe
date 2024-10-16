import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel files without headers, specifying all columns
jira_df = pd.read_excel('issues_openai_embeddings.xlsx', header=None)
feedback_df = pd.read_excel('feedback_openai_embeddings.xlsx', header=None)

# Function to concatenate the last 4 columns into a single numpy array
def convert_embedding_part(embedding_str):
    # Remove square brackets and split the string by commas
    embedding_str = embedding_str.strip('[]')
    embedding_parts = embedding_str.split(',')
    # Convert each part to float
    embedding_floats = [float(part) for part in embedding_parts if part.strip()]
    return embedding_floats

def concatenate_embeddings(row):
    # Initialize an empty list to hold the entire embedding
    full_embedding = []
    # Calculate the starting index for the last 4 columns
    start_index = len(row) - 4
    # Iterate over the last 4 columns to process each embedding part
    for i in range(start_index, len(row)):
        embedding_part = row.iloc[i]
        # Convert the embedding part to floats and extend the full embedding
        full_embedding.extend(convert_embedding_part(embedding_part))
    return np.array(full_embedding)



# Apply the function to concatenate embeddings for both DataFrames
jira_df['Embedding'] = jira_df.apply(concatenate_embeddings, axis=1)
feedback_df['Embedding'] = feedback_df.apply(concatenate_embeddings, axis=1)

# Initialize an empty DataFrame for the results
results_df = pd.DataFrame(columns=jira_df[0], index=feedback_df[0])

# Iterate over each row in the Jira DataFrame to compute cosine similarity
for jira_index, jira_row in jira_df.iterrows():
    jira_id = jira_row[0]
    jira_embedding = jira_row['Embedding']

    # Iterate over each row in the Feedback DataFrame
    for feedback_index, feedback_row in feedback_df.iterrows():
        feedback_id = feedback_row[0]
        feedback_embedding = feedback_row['Embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity([jira_embedding], [feedback_embedding])[0][0]

        # Assign the similarity score to the corresponding cell in the results DataFrame
        results_df.at[feedback_id, jira_id] = similarity

# Write the results to a new Excel file
results_df.to_excel('openai_similarity_scores.xlsx')
