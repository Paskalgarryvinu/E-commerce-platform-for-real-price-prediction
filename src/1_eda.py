# src/1_eda.py
import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
print(df.columns)
print(df['price'].describe())
print("missing images:", df['image_link'].isna().mean())
# price bins
print(pd.qcut(df['price'], q=10).value_counts())
# sample catalog contents:
print(df['catalog_content'].head(20).to_list())
