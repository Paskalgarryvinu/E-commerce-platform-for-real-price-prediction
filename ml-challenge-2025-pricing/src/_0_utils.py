# E:\project\src\_0_utils.py

import numpy as np
import re

# Fixed random seed for reproducibility
SEED = 42

# Symmetric Mean Absolute Percentage Error
def smape(y_true, y_pred):
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

# Extract item pack quantity (IPQ) from text like 'pack of 3' or '3 pcs'
def extract_ipq(text):
    if not isinstance(text, str):
        return 1
    text = text.lower()
    m = re.search(r'(\d+)\s*(?:pcs|pieces|count|pack|pk|x|×)', text)
    if m:
        return int(m.group(1))
    m2 = re.search(r'pack of\s*(\d+)', text)
    if m2:
        return int(m2.group(1))
    return 1
