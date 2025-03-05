from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Load the model
model = SentenceTransformer('./model/sentence-transformers_all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    # Encode texts into normalized embeddings
    embeddings = model.encode([text1, text2], normalize_embeddings=True)
    
    # Compute cosine similarity (dot product of normalized vectors)
    similarity = np.dot(embeddings[0], embeddings[1])
    
    # Ensure score is between 0 and 1
    score = max(0, similarity)
    return score

# Test with dataset
if __name__ == "__main__":
    dataset_path = "DataNeuron_Text_Similarity.csv"
    df = pd.read_csv(dataset_path)
    
    for index, row in df.head(5).iterrows():
        text1 = row['text1']
        text2 = row['text2']
        score = compute_similarity(text1, text2)
        print(f"Row {index + 1}:")
        print(f"Text1: {text1[:50]}...")  # Truncate for readability
        print(f"Text2: {text2[:50]}...")
        print(f"Similarity Score: {score:.4f}\n")