#!/usr/bin/env python3
"""
Create submission zip file for competition
"""

import os
import zipfile
import shutil
from datetime import datetime

def create_submission():
    """Create a zip file for competition submission."""
    print("=" * 60)
    print("CREATING SUBMISSION ZIP FILE")
    print("=" * 60)
    
    # Check if submission file exists
    submission_file = "outputs/sub_baseline_clean.csv"
    if not os.path.exists(submission_file):
        print("Submission file not found!")
        print("Run the pipeline first: python run_pipeline.py")
        return
    
    # Create zip file
    zip_filename = f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    print(f"Creating zip file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the main submission file
        zipf.write(submission_file, "sub_baseline_clean.csv")
        
        # Add any additional files you might want
        if os.path.exists("outputs/oof_baseline_clean.csv"):
            zipf.write("outputs/oof_baseline_clean.csv", "oof_baseline_clean.csv")
    
    # Get file size
    zip_size = os.path.getsize(zip_filename)
    
    print(f"\nZIP FILE CREATED SUCCESSFULLY!")
    print(f"   File name: {zip_filename}")
    print(f"   File size: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")
    print(f"   Location: {os.path.abspath(zip_filename)}")
    
    print(f"\nCONTENTS:")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file_info in zipf.filelist:
            print(f"   - {file_info.filename} ({file_info.file_size:,} bytes)")
    
    print(f"\nREADY FOR SUBMISSION!")
    print(f"   Upload this file to the competition portal:")
    print(f"   {zip_filename}")
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    create_submission()
