import pandas as pd
import spacy
from sklearn.decomposition import PCA
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import CosSimCalculator
from sklearn.feature_extraction.text import TfidfVectorizer

def get_filtered_embeddings(text):
    # Load spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    # Load DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # tokenize text with distilbert
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    # Calculate the contextualized token embeddings. that no gradients (backpropagation) is used
    with torch.no_grad():
        # unpack tokens into individual components
        outputs = model(**tokens)
    # represents the contextualized representation of each token in the input sequence.
    distilbert_embeddings = outputs.last_hidden_state
    #
    # Extract the token vectors only for nouns and verbs with spaCy
    # and convert to lowercase to make it case-insensitive.
    nouns_and_verbs = set(token.text.lower() for token in nlp(text) if token.pos_ in {"NOUN", "VERB"})
    relevant_embeddings = []

    for word in nouns_and_verbs:
        # tokenize word from nouns_and_verbs
        word_tokens = tokenizer.tokenize(word)
        # convert token-sequenz to String
        tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        # search for word in tokenized String, iterate through the tokenized text
        for i in range(len(tokenized_text) - len(word_tokens) + 1):
            # finds word_token in sequence of tokenized_text
            if tokenized_text[i:i + len(word_tokens)] == word_tokens:
                # The list embeddings then contains the embeddings for all identified nouns and verbs in the text.
                # These embeddings represent the context of each individual word in relation to its surrounding tokens.
                word_embedding = distilbert_embeddings[0, i:i + len(word_tokens), :].numpy()
                relevant_embeddings.extend(word_embedding)
    # This average (summary_embedding) then represents the context formed by the identified
    # nouns and verbs throughout the entire text.
    # It serves as a kind of summary of the context contributed by each individual word.
    if relevant_embeddings:
        average_embedding = np.mean(relevant_embeddings, axis=0)
    else:
        average_embedding = np.mean(distilbert_embeddings.squeeze().numpy(), axis=0)
    return average_embedding

def get_TFIDF_weighted_embedding(text, vectorizer):
    # Load spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    # Load DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # tokenize text with distilbert
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    # Calculate the contextualized token embeddings. that no gradients (backpropagation) is used
    with torch.no_grad():
        # unpack tokens into individual components
        outputs = model(**tokens)
    # represents the contextualized representation of each token in the input sequence.
    distilbert_embeddings = outputs.last_hidden_state
    #
    # Extract the token vectors only for nouns and verbs with spaCy
    # and convert to lowercase to make it case-insensitive.
    nouns_and_verbs = set(token.text.lower() for token in nlp(text) if token.pos_ in {"NOUN", "VERB"})

    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = {word: tfidf_matrix[0, feature_names.tolist().index(word)] if word in feature_names else 0 for word in nouns_and_verbs}

    weighted_embeddings = []

    for word, score in tfidf_scores.items():
        word_tokens = tokenizer.tokenize(word)
        tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

        for i in range(len(tokenized_text) - len(word_tokens) + 1):
            if tokenized_text[i:i + len(word_tokens)] == word_tokens:
                word_embedding = distilbert_embeddings[0, i:i + len(word_tokens), :].numpy()
                weighted_embedding = word_embedding * score
                weighted_embeddings.extend(weighted_embedding)
    if weighted_embeddings:
        average_embedding = np.mean(weighted_embeddings, axis=0)
    else:
        average_embedding = np.mean(distilbert_embeddings.squeeze().numpy(), axis=0)
    return average_embedding

def create_embeddings(file_path, output_path):
    print("Getting Embeddings")
    # Load the Excel file
    df = pd.read_excel(file_path, header=None)
    # Initialize a list to hold filtered embeddings
    embeddings_list = []
    # Process each row in the Excel file
    for index, row in df.iterrows():
        text = row[1]  # Assuming the text is in the second column
        filtered_embeddings = get_filtered_embeddings(text)
        embeddings_list.append(filtered_embeddings)
    # Create a new DataFrame with the original first column and filtered embeddings
    output_df = pd.DataFrame({
        'ID': df.iloc[:, 0],
        'Embeddings': embeddings_list
    })
    # Save the new DataFrame to an Excel file
    output_df.to_excel(output_path, index=False, header=False)


def create_combined_embeddings(issue_path, feedback_path, output_path, ground_truth_path):
    print("Getting Embeddings")
    # Load the Excel file
    issue_df = pd.read_excel(issue_path, header=None)
    feedback_df = pd.read_excel(feedback_path, header=None)

    ground_truth = CosSimCalculator.load_ground_truth(ground_truth_path)
    # Initialize a list to hold filtered embeddings
    embeddings_list = []
    # Process each row in the Excel file
    chosen_feedback = {}
    for index, issue in issue_df.iterrows():
        true_feedback_ids = ground_truth.get(issue[0], [])
        list_true_feedback_ids = list(true_feedback_ids)
        text = issue[1]
        if not text.endswith("."):
            text += "."
        if (len(list_true_feedback_ids) != 0):
            chosen_feedback[issue[0]] = list_true_feedback_ids[0]
            text += feedback_df[feedback_df.iloc[:, 0] == list_true_feedback_ids[0]].iloc[0, 1]
        filtered_embeddings = get_filtered_embeddings(text)
        embeddings_list.append(filtered_embeddings)
    # Create a new DataFrame with the original first column and filtered embeddings
    output_df = pd.DataFrame({
        'ID': issue_df.iloc[:, 0],
        'Embeddings': embeddings_list
    })
    # Save the new DataFrame to an Excel file
    output_df.to_excel(output_path, index=False, header=False)
    return chosen_feedback

def create_average_embeddings(issue_path, feedback_path, output_path, ground_truth_path, howmuchalreadyassigned=1):
    print("Getting Embeddings")
    # Load the Excel file
    issue_df = pd.read_excel(issue_path, header=None)
    feedback_df = pd.read_excel(feedback_path, header=None)

    ground_truth = CosSimCalculator.load_ground_truth(ground_truth_path)
    # Initialize a list to hold filtered embeddings
    embeddings_list = []
    # Process each row in the Excel file
    chosen_feedback = {}
    for index, issue in issue_df.iterrows():
        true_feedback_ids = ground_truth.get(issue[0], [])
        list_true_feedback_ids = list(true_feedback_ids)
        issue_text = issue[1]
        filtered_issue_embeddings = get_filtered_embeddings(issue_text)
        if (len(list_true_feedback_ids) != 0):
            selected_feedback_list = []
            filtered_feedback_embeddings= []
            for x in range(howmuchalreadyassigned):
                if(len(list_true_feedback_ids) > x):
                    feedback_text = feedback_df[feedback_df.iloc[:, 0] == list_true_feedback_ids[x]].iloc[0, 1]
                    selected_feedback_list.append(list_true_feedback_ids[x])
                    filtered_feedback_embeddings.append(get_filtered_embeddings(feedback_text))
            chosen_feedback[issue[0]] = selected_feedback_list
            filtered_embeddings = [filtered_issue_embeddings + fb for fb in filtered_feedback_embeddings]
            embeddings_list.append(np.mean(filtered_embeddings, axis=0))
        else:
            embeddings_list.append(filtered_issue_embeddings)
    # Create a new DataFrame with the original first column and filtered embeddings
    output_df = pd.DataFrame({
        'ID': issue_df.iloc[:, 0],
        'Embeddings': embeddings_list
    })
    # Save the new DataFrame to an Excel file
    output_df.to_excel(output_path, index=False, header=False)
    return chosen_feedback


def create_TFIDFweightedaverage_embeddings(issue_path, feedback_path, output_path, ground_truth_path):
    print("Getting Embeddings")
    # Load the Excel files
    issue_df = pd.read_excel(issue_path, header=None)
    feedback_df = pd.read_excel(feedback_path, header=None)

    ground_truth = CosSimCalculator.load_ground_truth(ground_truth_path)

    # Combine all texts to compute global TF-IDF
    all_texts = pd.concat([issue_df[1], feedback_df[1]]).values
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)

    # Initialize a list to hold filtered embeddings
    embeddings_list = []
    # Process each row in the Excel file
    chosen_feedback = {}
    for index, issue in issue_df.iterrows():
        true_feedback_ids = ground_truth.get(issue[0], [])
        list_true_feedback_ids = list(true_feedback_ids)
        issue_text = issue[1]
        filtered_issue_embeddings = get_TFIDF_weighted_embedding(issue_text,vectorizer)

        if len(list_true_feedback_ids) != 0:
            feedback_text = feedback_df[feedback_df.iloc[:, 0] == list_true_feedback_ids[0]].iloc[0, 1]
            chosen_feedback[issue[0]] = list_true_feedback_ids[0]
            filtered_feedback_embeddings = get_TFIDF_weighted_embedding(feedback_text,vectorizer)

            # Combine issue and feedback embeddings
            filtered_embeddings = (filtered_issue_embeddings + filtered_feedback_embeddings) / 2
            embeddings_list.append(filtered_embeddings)
        else:
            embeddings_list.append(filtered_issue_embeddings)

    # Create a new DataFrame with the original first column and filtered embeddings
    output_df = pd.DataFrame({
        'ID': issue_df.iloc[:, 0],
        'Embeddings': embeddings_list
    })
    # Save the new DataFrame to an Excel file
    output_df.to_excel(output_path, index=False, header=False)
    return chosen_feedback

def create_TFIDF_embeddings(embeddingsyouwant, othertext, output_path):
    print("Getting Embeddings")
    # Load the Excel file
    embeddingsyouwant_df = pd.read_excel(embeddingsyouwant, header=None)
    othertext_df = pd.read_excel(othertext, header=None)

    # Combine all texts to compute global TF-IDF
    all_texts = pd.concat([embeddingsyouwant_df[1], othertext_df[1]]).values
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)

    embeddings_list = []

    # Process each row in the Excel file
    for index, row in embeddingsyouwant_df.iterrows():
        text = row[1]  # Assuming the text is in the second column
        filtered_embeddings = get_TFIDF_weighted_embedding(text,vectorizer)
        embeddings_list.append(filtered_embeddings)
    # Create a new DataFrame with the original first column and filtered embeddings
    output_df = pd.DataFrame({
        'ID': embeddingsyouwant_df.iloc[:, 0],
        'Embeddings': embeddings_list
    })
    # Save the new DataFrame to an Excel file
    output_df.to_excel(output_path, index=False, header=False)

def create_concatenated_embeddings(issue_path, feedback_path, output_path, ground_truth_path, howmuchalreadyassigned=1, target_dim=768):
    print("Getting Embeddings")
    # Load the Excel file
    issue_df = pd.read_excel(issue_path, header=None)
    feedback_df = pd.read_excel(feedback_path, header=None)

    ground_truth = CosSimCalculator.load_ground_truth(ground_truth_path)
    # Initialize a list to hold filtered embeddings
    embeddings_list = []
    # Process each row in the Excel file
    chosen_feedback = {}
    for index, issue in issue_df.iterrows():
        true_feedback_ids = ground_truth.get(issue[0], [])
        list_true_feedback_ids = list(true_feedback_ids)
        issue_text = issue[1]
        filtered_issue_embeddings = get_filtered_embeddings(issue_text)
        if (len(list_true_feedback_ids) != 0):
            selected_feedback_list = []
            filtered_feedback_embeddings= [filtered_issue_embeddings]
            for x in range(howmuchalreadyassigned):
                if(len(list_true_feedback_ids) > x):
                    feedback_text = feedback_df[feedback_df.iloc[:, 0] == list_true_feedback_ids[x]].iloc[0, 1]
                    selected_feedback_list.append(list_true_feedback_ids[x])
                    filtered_feedback_embeddings.append(get_filtered_embeddings(feedback_text))
            chosen_feedback[issue[0]] = selected_feedback_list
            concatenated_embedding = np.concatenate(filtered_feedback_embeddings, axis=0)
            if len(concatenated_embedding) % target_dim != 0:
                print(concatenated_embedding.shape[0])
                raise ValueError("The length of the embedding must be a multiple of 768.")

            # Calculate the number of segments the embedding will be split into
            num_segments = len(concatenated_embedding) // target_dim

            # Reshape the embedding into a 2D array
            reshaped_embedding = concatenated_embedding.reshape(-1, target_dim)
            # Take the mean across the segments to reduce the embedding
            reduced_embedding = reshaped_embedding.mean(axis=0)
            embeddings_list.append(reduced_embedding)
        else:
            embeddings_list.append(filtered_issue_embeddings)
    # Create a new DataFrame with the original first column and filtered embeddings
    output_df = pd.DataFrame({
        'ID': issue_df.iloc[:, 0],
        'Embeddings': embeddings_list
    })
    # Save the new DataFrame to an Excel file
    output_df.to_excel(output_path, index=False, header=False)
    return chosen_feedback