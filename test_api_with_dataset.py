import requests
import pandas as pd

url = "http://localhost:5000/predict"

dataset_path = "DataNeuron_Text_Similarity.csv"
df = pd.read_csv(dataset_path)

for index, row in df.head(5).iterrows():
    data = {
        "text1": row['text1'],
        "text2": row['text2']
    }
    response = requests.post(url, json=data)
    result = response.json()
    print(f"Row {index + 1}:")
    print(f"Text1: {data['text1'][:50]}...")
    print(f"Text2: {data['text2'][:50]}...")
    print(f"Similarity Score: {result['similarity score']:.4f}\n")