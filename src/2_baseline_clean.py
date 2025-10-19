import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src._0_utils import smape, extract_ipq, SEED

# --- Configuration ---
DATA_DIR = "data/"
MODELS_DIR = "models/"
OUTPUTS_DIR = "outputs/"

# --- Load Data (Sample for faster execution) ---
print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}train.csv")
test = pd.read_csv(f"{DATA_DIR}test.csv")

# Use a sample for faster execution (10% of data)
print("Using 10% sample for faster execution...")
train_sample = train.sample(n=min(7500, len(train)), random_state=SEED)
print(f"Training on {len(train_sample)} samples instead of {len(train)}")

# --- Feature Engineering ---
print("Performing feature engineering...")
for df in [train_sample, test]:
    df['catalog_content'] = df['catalog_content'].fillna('')
    df['len_chars'] = df['catalog_content'].str.len()
    df['len_words'] = df['catalog_content'].str.split().map(len)
    df['num_digits'] = df['catalog_content'].str.count(r'\d')
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['has_image'] = df['image_link'].notna().astype(int)

# --- TF-IDF Vectorization (Reduced features) ---
print("Creating TF-IDF features...")
tf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_text_train = tf.fit_transform(train_sample['catalog_content'])
X_text_test = tf.transform(test['catalog_content'])

# --- Combine Features ---
print("Combining all features...")
num_features = ['len_chars', 'len_words', 'num_digits', 'ipq', 'has_image']
X_num_train = train_sample[num_features].fillna(0).values
X_num_test = test[num_features].fillna(0).values

X = hstack([X_text_train, csr_matrix(X_num_train)]).tocsr()
X_test = hstack([X_text_test, csr_matrix(X_num_test)]).tocsr()
y = train_sample['price'].values

# Transform the target variable to handle skewed price distribution
y_log = np.log1p(y)

# --- Cross-Validation and Model Training (Reduced folds) ---
print("Starting 3-fold cross-validation with LightGBM...")
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
oof = np.zeros(len(train_sample))
preds = np.zeros(len(test))

params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'learning_rate': 0.1,
    'num_leaves': 64,
    'seed': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    print(f"--- FOLD {fold + 1}/3 ---")
    
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr_log, y_val_log = y_log[tr_idx], y_log[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr_log)
    dval = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Predict on validation data and convert back from log scale
    val_preds_log = model.predict(X[val_idx], num_iteration=model.best_iteration)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    oof[val_idx] = val_preds

    # Predict on test data and convert back from log scale
    test_preds_log = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds_exp = np.expm1(test_preds_log)
    test_preds_exp[test_preds_exp < 0] = 0
    preds += test_preds_exp / kf.n_splits

    # Save the model from the last fold
    if fold == kf.n_splits - 1:
        joblib.dump(model, f"{MODELS_DIR}lgb_baseline_clean.pkl")

# --- Final Evaluation and Submission ---
print("\nCross-validation finished.")
final_smape = smape(y, oof)
print(f"Overall OOF SMAPE: {final_smape:.4f}")

# Save OOF predictions
train_sample['oof'] = oof
train_sample[['sample_id', 'oof']].to_csv(f"{OUTPUTS_DIR}oof_baseline_clean.csv", index=False)

# Save the final submission file
test['price'] = preds
test[['sample_id', 'price']].to_csv(f"{OUTPUTS_DIR}sub_baseline_clean.csv", index=False)

print(f"\nSubmission file created at: {OUTPUTS_DIR}sub_baseline_clean.csv")
print("Clean baseline script executed successfully!")





