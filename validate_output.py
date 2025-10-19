#!/usr/bin/env python3
"""
Validate that the output file meets competition requirements
"""

import pandas as pd
import os

def validate_output():
    """Check if output file meets all requirements."""
    print("=" * 60)
    print("OUTPUT VALIDATION FOR COMPETITION")
    print("=" * 60)
    
    # Check if files exist
    test_file = "data/test.csv"
    output_file = "outputs/sub_baseline_clean.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found!")
        return False
    
    if not os.path.exists(output_file):
        print("❌ Output file not found!")
        return False
    
    # Load data
    test = pd.read_csv(test_file)
    output = pd.read_csv(output_file)
    
    print(f"\nFILE COMPARISON:")
    print(f"   Test samples: {len(test):,}")
    print(f"   Output samples: {len(output):,}")
    
    # Requirement 1: Exactly 2 columns
    print(f"\nREQUIREMENT 1: 2 columns")
    has_2_columns = len(output.columns) == 2
    print(f"   Has 2 columns: {has_2_columns}")
    if has_2_columns:
        print(f"   Columns: {list(output.columns)}")
    
    # Requirement 2: Columns are sample_id and price
    print(f"\nREQUIREMENT 2: Correct column names")
    correct_columns = list(output.columns) == ['sample_id', 'price']
    print(f"   Correct columns: {correct_columns}")
    
    # Requirement 3: All sample IDs present
    print(f"\nREQUIREMENT 3: All sample IDs present")
    all_samples_present = len(test) == len(output)
    print(f"   All samples present: {all_samples_present}")
    
    # Requirement 4: Sample IDs match exactly
    print(f"\nREQUIREMENT 4: Sample IDs match")
    sample_ids_match = test['sample_id'].equals(output['sample_id'])
    print(f"   Sample IDs match: {sample_ids_match}")
    
    # Requirement 5: sample_id is integer
    print(f"\nREQUIREMENT 5: sample_id data type")
    sample_id_int = output['sample_id'].dtype == 'int64'
    print(f"   sample_id is integer: {sample_id_int}")
    print(f"   sample_id type: {output['sample_id'].dtype}")
    
    # Requirement 6: price is float
    print(f"\nREQUIREMENT 6: price data type")
    price_float = output['price'].dtype == 'float64'
    print(f"   price is float: {price_float}")
    print(f"   price type: {output['price'].dtype}")
    
    # Requirement 7: No missing values
    print(f"\nREQUIREMENT 7: No missing values")
    no_missing = output.isnull().sum().sum() == 0
    print(f"   No missing values: {no_missing}")
    if not no_missing:
        print(f"   Missing values: {output.isnull().sum().to_dict()}")
    
    # Additional checks
    print(f"\nADDITIONAL INFO:")
    print(f"   Price range: ${output['price'].min():.2f} - ${output['price'].max():.2f}")
    print(f"   Average price: ${output['price'].mean():.2f}")
    print(f"   File size: {os.path.getsize(output_file):,} bytes")
    
    # Final verdict
    all_requirements_met = all([
        has_2_columns,
        correct_columns,
        all_samples_present,
        sample_ids_match,
        sample_id_int,
        price_float,
        no_missing
    ])
    
    print(f"\n{'='*60}")
    if all_requirements_met:
        print("SUCCESS: Your output file meets ALL requirements!")
        print("Ready for competition submission!")
    else:
        print("FAILED: Some requirements not met!")
        print("Please fix the issues above before submission.")
    print(f"{'='*60}")
    
    return all_requirements_met

if __name__ == "__main__":
    validate_output()
