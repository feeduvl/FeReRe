import pandas as pd
import spacy
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

def get_filtered_embeddings(text):
    # Load spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    # Load DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # tokenize text with distilbert
    tokens = tokenizer(text, return_tensors="pt")
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

def process_excel(file_path, output_path):

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

