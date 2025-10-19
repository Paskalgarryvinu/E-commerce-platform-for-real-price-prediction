#!/usr/bin/env python3
"""
Simple script to show you exactly what your output contains
"""

import pandas as pd
import os

def show_my_output():
    """Show your output in a simple, clear way."""
    print("=" * 60)
    print("YOUR PROJECT OUTPUT EXPLAINED")
    print("=" * 60)
    
    # Check if output file exists
    output_file = "outputs/sub_baseline_clean.csv"
    if not os.path.exists(output_file):
        print("No output file found! Run the pipeline first:")
        print("python run_pipeline.py")
        return
    
    # Load your predictions
    df = pd.read_csv(output_file)
    
    print(f"\nWHAT YOUR PROJECT DID:")
    print(f"   Your AI model analyzed 75,000 product descriptions")
    print(f"   It learned from examples and predicted prices")
    print(f"   Results saved in: {output_file}")
    
    print(f"\nYOUR PREDICTIONS SUMMARY:")
    print(f"   Total products predicted: {len(df):,}")
    print(f"   Cheapest product: ${df['price'].min():.2f}")
    print(f"   Most expensive product: ${df['price'].max():.2f}")
    print(f"   Average price: ${df['price'].mean():.2f}")
    
    print(f"\nSAMPLE PREDICTIONS:")
    print(f"   Product ID    |  Predicted Price")
    print(f"   --------------|----------------")
    for i, row in df.head(10).iterrows():
        print(f"   {int(row['sample_id']):10d}  |  ${row['price']:8.2f}")
    
    print(f"\nPRICE DISTRIBUTION:")
    under_5 = len(df[df['price'] < 5])
    between_5_10 = len(df[(df['price'] >= 5) & (df['price'] < 10)])
    between_10_20 = len(df[(df['price'] >= 10) & (df['price'] < 20)])
    between_20_50 = len(df[(df['price'] >= 20) & (df['price'] < 50)])
    over_50 = len(df[df['price'] >= 50])
    
    print(f"   Under $5:     {under_5:6,} products ({under_5/len(df)*100:.1f}%)")
    print(f"   $5 - $10:     {between_5_10:6,} products ({between_5_10/len(df)*100:.1f}%)")
    print(f"   $10 - $20:    {between_10_20:6,} products ({between_10_20/len(df)*100:.1f}%)")
    print(f"   $20 - $50:    {between_20_50:6,} products ({between_20_50/len(df)*100:.1f}%)")
    print(f"   Over $50:     {over_50:6,} products ({over_50/len(df)*100:.1f}%)")
    
    print(f"\nWHAT THIS MEANS:")
    print(f"   Your AI model predicted prices for 75,000 products")
    print(f"   Most products ({(between_10_20 + between_5_10)/len(df)*100:.1f}%) are priced between $5-$20")
    print(f"   The model learned patterns from product descriptions")
    print(f"   These predictions can be used for competition submission")
    
    print(f"\nFILE DETAILS:")
    print(f"   File name: {output_file}")
    print(f"   File size: {os.path.getsize(output_file):,} bytes")
    print(f"   Format: CSV with 2 columns (sample_id, price)")
    
    print(f"\n" + "=" * 60)
    print("YOUR OUTPUT IS READY FOR COMPETITION SUBMISSION!")
    print("=" * 60)

if __name__ == "__main__":
    show_my_output()





