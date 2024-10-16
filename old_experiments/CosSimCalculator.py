import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def convert_embedding_part(embedding_str):
    # Remove square brackets and split the string by commas
    embedding_str = embedding_str.strip('[]')
    embedding_parts = embedding_str.split(',')
    embedding_parts = embedding_str.split()
    # Convert each part to float
    embedding_floats = [float(part) for part in embedding_parts if part.strip()]
    return embedding_floats

def process_last_embedding(row):
    # Access the last column directly
    last_embedding_part = row.iloc[-1]
    # Convert the last embedding part to floats
    embedding_floats = convert_embedding_part(last_embedding_part)
    # Return the embedding as a numpy array
    return np.array(embedding_floats)

def calc_cos_sim(issue_emb_path, feedback_emb_path, chosen_feedback=None):
    print("Calculating Cosine Similarity")
    # Load the Excel files without headers, specifying all columns
    jira_df = pd.read_excel(issue_emb_path, header=None)
    feedback_df = pd.read_excel(feedback_emb_path, header=None)

    # Apply the function to concatenate embeddings for both DataFrames
    jira_df['Embedding'] = jira_df.apply(process_last_embedding, axis=1)
    feedback_df['Embedding'] = feedback_df.apply(process_last_embedding, axis=1)

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
            similarity = cosine_similarity([jira_embedding], [feedback_embedding])[0][0]
            if (chosen_feedback != None and jira_id in chosen_feedback and feedback_id == chosen_feedback[jira_id]):
                similarity = np.NaN

            # Assign the similarity score to the corresponding cell in the results DataFrame
            results_df.at[feedback_id, jira_id] = similarity

    # Write the results to a new Excel file
    results_df.to_excel('data/bert/bert_similarity_scores.xlsx')

def load_ground_truth(path):
    df = pd.read_excel(path, header=None)
    ground_truth = {}
    for column in df:
        issue_id = df.iloc[0, column]
        feedback_ids = set(df.iloc[1:, column].dropna().astype(str))
        ground_truth[str(issue_id)] = feedback_ids  # Ensure issue IDs are strings for consistent comparison
    return ground_truth

def calc_cos_sim_incl_one_feedback_highestscore(issue_emb_path, feedback_emb_path):
    print("Calculating Cosine Similarity")
    # Load the Excel files without headers, specifying all columns
    jira_df = pd.read_excel(issue_emb_path, header=None)
    feedback_df = pd.read_excel(feedback_emb_path, header=None)

    # Apply the function to concatenate embeddings for both DataFrames
    jira_df['Embedding'] = jira_df.apply(process_last_embedding, axis=1)
    feedback_df['Embedding'] = feedback_df.apply(process_last_embedding, axis=1)
    ground_truth = load_ground_truth('data/Ground_Truth.xlsx')

    results_df = pd.DataFrame(columns=jira_df[0], index=feedback_df[0])
    chosen_feedback={}
    for issue_index,issue in jira_df.iterrows():
        issue_key=issue[0]
        issue_emb=issue['Embedding']
        true_feedback_ids=ground_truth.get(issue_key, [])
        true_feedback_embeddings = {}
        for feedback_id in true_feedback_ids:
            # Find the row in feedback_df where the ID matches feedback_id
            row = feedback_df[feedback_df[0] == feedback_id]
            # Extract the embedding vector from the row
            if not row.empty:
                embedding = row['Embedding'].iloc[0]  # Adjust column names as per your Excel structure
                true_feedback_embeddings[feedback_id]=embedding
            else:
                print(f"{feedback_id} not found.")

        list_true_feedback_ids = list(true_feedback_ids)
        if(len(list_true_feedback_ids)!=0):
            chosen_feedback[issue_key]=list_true_feedback_ids[0]
        for feedback_index, feedback_row in feedback_df.iterrows():
            feedback_id = feedback_row[0]
            feedback_embedding = feedback_row['Embedding']
            # Calculate cosine similarity
            if_similarity = cosine_similarity([issue_emb], [feedback_embedding])[0][0]
            ff_similarity = np.NaN

            if(len(list_true_feedback_ids)!=0) and list_true_feedback_ids[0] != feedback_id:
                ff_similarity = cosine_similarity([true_feedback_embeddings[list_true_feedback_ids[0]]],[feedback_embedding])[0][0]
            if ff_similarity != np.NaN and ff_similarity > if_similarity:
                similarity = ff_similarity
            else:
                similarity = if_similarity
            # Assign the similarity score to the corresponding cell in the results DataFrame
            results_df.at[feedback_id, issue_key] = similarity

    # Write the results to a new Excel file
    results_df.to_excel('data/bert/bert_similarity_scores.xlsx')
    return chosen_feedback

def calc_cos_sim_incl_one_feedback_avgcossim(issue_emb_path, feedback_emb_path):
    print("Calculating Cosine Similarity")
    # Load the Excel files without headers, specifying all columns
    jira_df = pd.read_excel(issue_emb_path, header=None)
    feedback_df = pd.read_excel(feedback_emb_path, header=None)

    # Apply the function to concatenate embeddings for both DataFrames
    jira_df['Embedding'] = jira_df.apply(process_last_embedding, axis=1)
    feedback_df['Embedding'] = feedback_df.apply(process_last_embedding, axis=1)
    ground_truth = load_ground_truth('data/Ground_Truth.xlsx')

    results_df = pd.DataFrame(columns=jira_df[0], index=feedback_df[0])
    chosen_feedback={}
    for issue_index,issue in jira_df.iterrows():
        issue_key=issue[0]
        issue_emb=issue['Embedding']
        true_feedback_ids=ground_truth.get(issue_key, [])
        true_feedback_embeddings = {}
        for feedback_id in true_feedback_ids:
            # Find the row in feedback_df where the ID matches feedback_id
            row = feedback_df[feedback_df[0] == feedback_id]
            # Extract the embedding vector from the row
            if not row.empty:
                embedding = row['Embedding'].iloc[0]  # Adjust column names as per your Excel structure
                true_feedback_embeddings[feedback_id]=embedding
            else:
                print(f"{feedback_id} not found.")

        list_true_feedback_ids = list(true_feedback_ids)
        if(len(list_true_feedback_ids)!=0):
            chosen_feedback[issue_key]=list_true_feedback_ids[0]
        for feedback_index, feedback_row in feedback_df.iterrows():
            feedback_id = feedback_row[0]
            feedback_embedding = feedback_row['Embedding']
            # Calculate cosine similarity
            if_similarity = cosine_similarity([issue_emb], [feedback_embedding])[0][0]
            ff_similarity = np.NaN

            if(len(list_true_feedback_ids)!=0) and list_true_feedback_ids[0] != feedback_id:
                ff_similarity = cosine_similarity([true_feedback_embeddings[list_true_feedback_ids[0]]],[feedback_embedding])[0][0]
            #Calculate average similarity score
            similarity = (if_similarity + ff_similarity) / 2
            # Assign the similarity score to the corresponding cell in the results DataFrame
            results_df.at[feedback_id, issue_key] = similarity

    # Write the results to a new Excel file
    results_df.to_excel('data/bert/bert_similarity_scores.xlsx')
    return chosen_feedback





