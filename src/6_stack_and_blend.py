import numpy as np, pandas as pd
# assume model_preds = {'tfidf': preds1, 'text_emb': preds2, 'img': preds3}
oof_smape = {'tfidf': 8.0, 'text_emb': 6.5, 'img': 7.0}  # example
weights = {k: 1.0/v for k,v in oof_smape.items()}
total = sum(weights.values())
weights = {k: w/total for k,w in weights.items()}
final_pred = sum(weights[k]*model_preds[k] for k in model_preds)
