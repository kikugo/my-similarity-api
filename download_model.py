from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./model/sentence-transformers_all-MiniLM-L6-v2')