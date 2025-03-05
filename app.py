from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

model = SentenceTransformer('./model/sentence-transformers_all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    embeddings = model.encode([text1, text2], normalize_embeddings=True)
    similarity = np.dot(embeddings[0], embeddings[1])
    score = max(0, similarity)
    return score

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text1 = data['text1']
        text2 = data['text2']
        
        score = compute_similarity(text1, text2)
        
        return jsonify({'similarity score': score})
    except KeyError:
        return jsonify({'error': 'Missing text1 or text2 in request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)