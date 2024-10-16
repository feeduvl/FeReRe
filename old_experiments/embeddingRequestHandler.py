import pandas as pd
import requests

def fetch_embeddings(text_list):
    key= input("API key: ")
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": "Bearer "+key,  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-large",
        "input": text_list
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if 'data' in response_data:
                return [item['embedding'] for item in response_data['data']]
            else:
                print(f"No embeddings found. Response: {response_data}")
                return [None] * len(text_list)
        else:
            print(f"API request failed. Status Code: {response.status_code}. Response: {response.text}")
            return [None] * len(text_list)
    except Exception as e:
        print(f"Error fetching embeddings, error: {e}")
        return [None] * len(text_list)

def split_embedding(embedding, max_size=1000):
    """Split the embedding into chunks of max_size."""
    return [embedding[i:i + max_size] for i in range(0, len(embedding), max_size)]

def main():
    file_path = 'feedback_output.xlsx'  # Update this to your file path

    # Load the Excel file without assuming the first row as a header
    df = pd.read_excel(file_path, header=None)
    # Use the second column for generating embeddings
    embeddings_list = fetch_embeddings(df.iloc[:, 1].astype(str).tolist())

    # Split large embeddings across multiple cells
    max_embedding_size = 1000  # Define max size for each part of the split embedding
    split_embeddings = [split_embedding(embedding, max_size=max_embedding_size) for embedding in embeddings_list]

    # The largest number of splits determines the number of columns needed
    max_splits = max(len(splits) for splits in split_embeddings)

    # Create new columns for each part of the split embeddings
    for i in range(max_splits):
        df[f'Embedding_{i+1}'] = pd.Series([splits[i] if i < len(splits) else [] for splits in split_embeddings])

    # Save the result in a new Excel file, including the first and second columns
    output_file_pat = 'feedback_openai_embeddings.xlsx'
    df.to_excel(output_file_path, index=False, header=None)

    print(f"Completed. The output is saved to {output_file_path}")

if __name__ == "__main__":
    main()
