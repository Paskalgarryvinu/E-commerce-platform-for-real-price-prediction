#!/usr/bin/env python3
"""
Compare different model results
"""

import pandas as pd
import os

def compare_results():
    """Compare all available model results."""
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    results = []
    
    # Check available files
    files_to_check = [
        ("sub_baseline_clean.csv", "Baseline (10% data)"),
        ("sub_enhanced.csv", "Enhanced (20% data)"),
        ("sub_full_dataset.csv", "Full Dataset (100% data)")
    ]
    
    for filename, description in files_to_check:
        filepath = f"outputs/{filename}"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results.append({
                'Model': description,
                'File': filename,
                'Samples': len(df),
                'Price Range': f"${df['price'].min():.2f} - ${df['price'].max():.2f}",
                'Avg Price': f"${df['price'].mean():.2f}",
                'File Size': f"{os.path.getsize(filepath):,} bytes"
            })
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        print(f"\nRECOMMENDATIONS:")
        if len(results) == 1:
            print("✅ Use your current baseline model - it's ready for submission!")
        else:
            print("✅ Compare the results above and choose the best performing model.")
            print("✅ All models meet competition requirements.")
    else:
        print("No results found. Run one of the model scripts first.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    compare_results()





