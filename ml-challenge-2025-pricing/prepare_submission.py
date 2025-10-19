#!/usr/bin/env python3
"""
Prepare the exact submission file format for competition portal
"""

import pandas as pd
import os

def prepare_submission():
    """Create test_out.csv in the exact format required."""
    print("=" * 60)
    print("PREPARING COMPETITION SUBMISSION FILE")
    print("=" * 60)
    
    # Check if our predictions exist
    pred_file = "outputs/sub_baseline_clean.csv"
    if not os.path.exists(pred_file):
        print("Prediction file not found!")
        print("Run the pipeline first: python run_pipeline.py")
        return
    
    # Load our predictions
    print("Loading predictions...")
    predictions = pd.read_csv(pred_file)
    
    # Create the exact format they want
    print("Creating test_out.csv...")
    
    # Rename the file to match their requirement
    submission_file = "test_out.csv"
    predictions.to_csv(submission_file, index=False)
    
    # Verify the format
    print("Verifying format...")
    df_check = pd.read_csv(submission_file)
    
    print(f"\nSUBMISSION FILE CREATED!")
    print(f"   File name: {submission_file}")
    print(f"   Total predictions: {len(df_check):,}")
    print(f"   Columns: {list(df_check.columns)}")
    print(f"   File size: {os.path.getsize(submission_file):,} bytes")
    
    # Show sample
    print(f"\nSAMPLE OF YOUR SUBMISSION:")
    print(df_check.head(10).to_string(index=False))
    
    print(f"\nSUBMISSION CHECKLIST:")
    print(f"   File name: test_out.csv")
    print(f"   Format: CSV")
    print(f"   Columns: sample_id, price")
    print(f"   Total rows: {len(df_check):,}")
    print(f"   All prices positive: {(df_check['price'] > 0).all()}")
    print(f"   No missing values: {df_check.isnull().sum().sum() == 0}")
    
    print(f"\nREADY FOR UPLOAD!")
    print(f"   Upload this file to the competition portal:")
    print(f"   {os.path.abspath(submission_file)}")
    
    print(f"\n" + "=" * 60)
    print("SUBMISSION FILE READY!")
    print("=" * 60)
    
    return submission_file

if __name__ == "__main__":
    prepare_submission()
