#!/usr/bin/env python3
"""
Quick Pipeline Runner for Price Prediction Project
This script runs the essential parts of the ML pipeline in order.
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and report the results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS - Completed in {duration:.2f} seconds")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Show last 500 chars
        else:
            print(f"FAILED - Error after {duration:.2f} seconds")
            print("Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT - Script took longer than 5 minutes")
    except Exception as e:
        print(f"EXCEPTION - {str(e)}")

def main():
    """Run the ML pipeline in sequence."""
    print("Starting Price Prediction Pipeline")
    print("This will run the essential scripts for the ML project")
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the pipeline steps
    pipeline_steps = [
        ("src/1_eda.py", "Exploratory Data Analysis"),
        ("src/2_baseline_clean.py", "Clean Baseline Model (TF-IDF + LightGBM)"),
    ]
    
    # Run each step
    for script, description in pipeline_steps:
        if os.path.exists(script):
            run_script(script, description)
        else:
            print(f"WARNING: Script not found: {script}")
    
    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print("Check the 'outputs/' directory for results:")
    
    # List output files
    if os.path.exists("outputs"):
        output_files = os.listdir("outputs")
        for file in output_files:
            if file.endswith('.csv'):
                file_path = os.path.join("outputs", file)
                size = os.path.getsize(file_path)
                print(f"  {file} ({size:,} bytes)")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
