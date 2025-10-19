# src/3_text_embeddings.py
from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np
model = SentenceTransformer('all-mpnet-base-v2')   # moderate size
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
texts = train['catalog_content'].fillna('').tolist()
embs = model.encode(texts, show_progress_bar=True, batch_size=64)
np.save("../models/embeddings_text_train.npy", embs)
# test embeddings
test_embs = model.encode(test['catalog_content'].fillna('').tolist(), batch_size=64, show_progress_bar=True)
np.save("../models/embeddings_text_test.npy", test_embs)
