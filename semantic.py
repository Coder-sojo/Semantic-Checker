# semantic_checker.py
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch
model = SentenceTransformer('all-MiniLM-L6-v2')
def embed_text(text):
    return model.encode(text, convert_to_tensor=True)
def compare_meaning(text1, text2):
    vec1 = embed_text(text1)
    vec2 = embed_text(text2)
    similarity = cosine_similarity(vec1, vec2, dim=0)
    return similarity.item()
if __name__ == "__main__":
    print("Semantic Clause Checker")
    sentence1 = input("Enter SBD Clause: ")
    sentence2 = input("Enter Proposal Clause: ")
    score = compare_meaning(sentence1, sentence2)
    print("\n Semantic Similarity Score:", round(score, 4))
    if score > 0.85:
        print("Likely a full match.")
    elif score > 0.65:
        print("Possibly a partial match.")
    else:
        print("Likely not matching.")
