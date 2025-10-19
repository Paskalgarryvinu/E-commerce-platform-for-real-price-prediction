#!/usr/bin/env python3
"""
Simple script to view your ML model results
"""

import pandas as pd
import os

def view_predictions():
    """Display your model predictions in a nice format."""
    print("=" * 60)
    print("YOUR MACHINE LEARNING RESULTS")
    print("=" * 60)
    
    # Check if prediction file exists
    pred_file = "outputs/sub_baseline_clean.csv"
    if not os.path.exists(pred_file):
        print("No predictions found! Run the pipeline first:")
        print("   python run_pipeline.py")
        return
    
    # Load predictions
    df = pd.read_csv(pred_file)
    
    print(f"\nPREDICTION SUMMARY:")
    print(f"   Total predictions: {len(df):,}")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"   Average price: ${df['price'].mean():.2f}")
    print(f"   Median price: ${df['price'].median():.2f}")
    
    print(f"\nFIRST 20 PREDICTIONS:")
    print("-" * 40)
    for i, row in df.head(20).iterrows():
        print(f"   Sample {int(row['sample_id']):6d}: ${row['price']:8.2f}")
    
    print(f"\nLAST 10 PREDICTIONS:")
    print("-" * 40)
    for i, row in df.tail(10).iterrows():
        print(f"   Sample {int(row['sample_id']):6d}: ${row['price']:8.2f}")
    
    # Show price distribution
    print(f"\nPRICE DISTRIBUTION:")
    print("-" * 40)
    price_ranges = [
        (0, 5, "Under $5"),
        (5, 10, "$5 - $10"),
        (10, 20, "$10 - $20"),
        (20, 50, "$20 - $50"),
        (50, 100, "$50 - $100"),
        (100, float('inf'), "Over $100")
    ]
    
    for min_price, max_price, label in price_ranges:
        count = len(df[(df['price'] >= min_price) & (df['price'] < max_price)])
        percentage = (count / len(df)) * 100
        print(f"   {label:12s}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\nYour submission file is ready: {pred_file}")
    print(f"   File size: {os.path.getsize(pred_file):,} bytes")
    
    print("\n" + "=" * 60)

def view_training_results():
    """Display training/cross-validation results."""
    oof_file = "outputs/oof_baseline_clean.csv"
    if os.path.exists(oof_file):
        print(f"\nCROSS-VALIDATION RESULTS:")
        print("-" * 40)
        oof_df = pd.read_csv(oof_file)
        print(f"   Cross-validation samples: {len(oof_df):,}")
        print(f"   OOF predictions range: ${oof_df['oof'].min():.2f} - ${oof_df['oof'].max():.2f}")
        print(f"   Average OOF prediction: ${oof_df['oof'].mean():.2f}")

if __name__ == "__main__":
    view_predictions()
    view_training_results()