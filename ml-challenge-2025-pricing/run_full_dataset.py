#!/usr/bin/env python3
"""
Run the model on the FULL dataset (75,000 samples) for maximum performance
WARNING: This will take much longer (30-60 minutes)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src._0_utils import smape, extract_ipq, SEED

def run_full_dataset():
    """Run model on full dataset."""
    print("=" * 60)
    print("RUNNING FULL DATASET MODEL")
    print("WARNING: This will take 30-60 minutes!")
    print("=" * 60)
    
    # Load data
    print("Loading FULL dataset...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    print(f"Training on ALL {len(train):,} samples")
    
    # Feature engineering
    print("Feature engineering...")
    for df in [train, test]:
        df['catalog_content'] = df['catalog_content'].fillna('')
        df['len_chars'] = df['catalog_content'].str.len()
        df['len_words'] = df['catalog_content'].str.split().map(len)
        df['num_digits'] = df['catalog_content'].str.count(r'\d')
        df['ipq'] = df['catalog_content'].apply(extract_ipq)
        df['has_image'] = df['image_link'].notna().astype(int)
    
    # TF-IDF
    print("Creating TF-IDF features...")
    tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
    X_text_train = tf.fit_transform(train['catalog_content'])
    X_text_test = tf.transform(test['catalog_content'])
    
    # Combine features
    print("Combining features...")
    num_features = ['len_chars', 'len_words', 'num_digits', 'ipq', 'has_image']
    X_num_train = train[num_features].fillna(0).values
    X_num_test = test[num_features].fillna(0).values
    
    X = hstack([X_text_train, csr_matrix(X_num_train)]).tocsr()
    X_test = hstack([X_text_test, csr_matrix(X_num_test)]).tocsr()
    y = train['price'].values
    y_log = np.log1p(y)
    
    # Model training
    print("Training model on FULL dataset...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"--- FOLD {fold + 1}/5 ---")
        
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr_log, y_val_log = y_log[tr_idx], y_log[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr_log)
        dval = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        # Predictions
        val_preds_log = model.predict(X[val_idx], num_iteration=model.best_iteration)
        val_preds = np.expm1(val_preds_log)
        val_preds[val_preds < 0] = 0
        oof[val_idx] = val_preds
        
        test_preds_log = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds_exp = np.expm1(test_preds_log)
        test_preds_exp[test_preds_exp < 0] = 0
        preds += test_preds_exp / kf.n_splits
    
    # Results
    print("\nFull dataset results:")
    final_smape = smape(y, oof)
    print(f"Full dataset OOF SMAPE: {final_smape:.4f}")
    
    # Save predictions
    test['price'] = preds
    test[['sample_id', 'price']].to_csv("outputs/sub_full_dataset.csv", index=False)
    
    print(f"\nFull dataset submission file created: outputs/sub_full_dataset.csv")
    print("Full dataset model completed!")
    
    return final_smape

if __name__ == "__main__":
    run_full_dataset()





