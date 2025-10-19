#!/usr/bin/env python3
"""
Script to improve your model performance
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

def improve_model():
    """Create an improved version of the model."""
    print("=" * 60)
    print("IMPROVING YOUR MODEL")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    # Use 20% of data for better performance (instead of 10%)
    print("Using 20% sample for better performance...")
    train_sample = train.sample(n=min(15000, len(train)), random_state=SEED)
    print(f"Training on {len(train_sample)} samples")
    
    # Enhanced feature engineering
    print("Enhanced feature engineering...")
    for df in [train_sample, test]:
        df['catalog_content'] = df['catalog_content'].fillna('')
        
        # Basic features
        df['len_chars'] = df['catalog_content'].str.len()
        df['len_words'] = df['catalog_content'].str.split().map(len)
        df['num_digits'] = df['catalog_content'].str.count(r'\d')
        df['ipq'] = df['catalog_content'].apply(extract_ipq)
        df['has_image'] = df['image_link'].notna().astype(int)
        
        # Additional features
        df['num_uppercase'] = df['catalog_content'].str.count(r'[A-Z]')
        df['num_special'] = df['catalog_content'].str.count(r'[!@#$%^&*(),.?":{}|<>]')
        df['has_bullet_points'] = df['catalog_content'].str.contains('Bullet Point').astype(int)
        df['has_value_unit'] = df['catalog_content'].str.contains('Value:|Unit:').astype(int)
        df['word_density'] = df['len_words'] / (df['len_chars'] + 1)
    
    # Enhanced TF-IDF
    print("Creating enhanced TF-IDF features...")
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), min_df=2)  # More features
    X_text_train = tf.fit_transform(train_sample['catalog_content'])
    X_text_test = tf.transform(test['catalog_content'])
    
    # Combine all features
    print("Combining all features...")
    num_features = ['len_chars', 'len_words', 'num_digits', 'ipq', 'has_image',
                   'num_uppercase', 'num_special', 'has_bullet_points', 
                   'has_value_unit', 'word_density']
    
    X_num_train = train_sample[num_features].fillna(0).values
    X_num_test = test[num_features].fillna(0).values
    
    X = hstack([X_text_train, csr_matrix(X_num_train)]).tocsr()
    X_test = hstack([X_text_test, csr_matrix(X_num_test)]).tocsr()
    y = train_sample['price'].values
    y_log = np.log1p(y)
    
    # Enhanced model training
    print("Training enhanced model...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)  # 5 folds instead of 3
    oof = np.zeros(len(train_sample))
    preds = np.zeros(len(test))
    
    # Better parameters
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'learning_rate': 0.05,  # Lower learning rate
        'num_leaves': 128,      # More leaves
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
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
            num_boost_round=1000,  # More rounds
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
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
    print("\nEnhanced model results:")
    final_smape = smape(y, oof)
    print(f"Enhanced OOF SMAPE: {final_smape:.4f}")
    
    # Save enhanced predictions
    test['price'] = preds
    test[['sample_id', 'price']].to_csv("outputs/sub_enhanced.csv", index=False)
    
    print(f"\nEnhanced submission file created: outputs/sub_enhanced.csv")
    print("Enhanced model completed!")
    
    return final_smape

if __name__ == "__main__":
    improve_model()





