# src/4_image_embeddings.py
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os, numpy as np, pandas as pd
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df = pd.read_csv("../data/train.csv")
embs = []
for _, r in tqdm(df.iterrows(), total=len(df)):
    img_path = f"../data/images_cache/{r['sample_id']}.jpg"
    if not os.path.exists(img_path):
        embs.append(np.zeros(512))
        continue
    image = Image.open(img_path).convert('RGB').resize((224,224))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**inputs)
        image_emb = image_emb.cpu().numpy().reshape(-1)
    embs.append(image_emb)
embs = np.vstack(embs)
np.save("../models/embeddings_image_train.npy", embs)
# repeat for test
