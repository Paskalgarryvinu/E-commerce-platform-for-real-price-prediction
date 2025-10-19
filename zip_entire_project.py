#!/usr/bin/env python3
"""
Create a zip file containing the entire project
"""

import os
import zipfile
import shutil
from datetime import datetime

def zip_entire_project():
    """Create a zip file with the entire project."""
    print("=" * 60)
    print("CREATING ENTIRE PROJECT ZIP FILE")
    print("=" * 60)
    
    # Create zip file with timestamp
    zip_filename = f"complete_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    print(f"Creating complete project zip: {zip_filename}")
    print("This may take a few minutes...")
    
    # Files and folders to include
    items_to_zip = [
        "src/",           # Source code
        "data/",          # Dataset
        "outputs/",       # Results
        "models/",        # Model files
        "*.py",           # Python scripts
        "requirements.txt" # Dependencies
    ]
    
    # Files to exclude (to keep zip size reasonable)
    exclude_patterns = [
        "venv/",          # Virtual environment (too large)
        "__pycache__/",   # Python cache
        "*.pyc",          # Compiled Python files
        ".git/",          # Git files
        "*.log"           # Log files
    ]
    
    total_size = 0
    file_count = 0
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all Python files in root
        for file in os.listdir('.'):
            if file.endswith('.py') and os.path.isfile(file):
                zipf.write(file)
                total_size += os.path.getsize(file)
                file_count += 1
                print(f"   Added: {file}")
        
        # Add requirements.txt if exists
        if os.path.exists('requirements.txt'):
            zipf.write('requirements.txt')
            total_size += os.path.getsize('requirements.txt')
            file_count += 1
            print(f"   Added: requirements.txt")
        
        # Add src/ directory
        if os.path.exists('src'):
            for root, dirs, files in os.walk('src'):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    if not file.endswith('.pyc'):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path)
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                        print(f"   Added: {file_path}")
        
        # Add outputs/ directory (but limit size)
        if os.path.exists('outputs'):
            for file in os.listdir('outputs'):
                if file.endswith('.csv'):
                    file_path = os.path.join('outputs', file)
                    zipf.write(file_path)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                    print(f"   Added: {file_path}")
        
        # Add models/ directory
        if os.path.exists('models'):
            for file in os.listdir('models'):
                file_path = os.path.join('models', file)
                if os.path.isfile(file_path):
                    zipf.write(file_path)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                    print(f"   Added: {file_path}")
        
        # Add sample data files (first few rows only to keep size reasonable)
        if os.path.exists('data'):
            for file in os.listdir('data'):
                if file.endswith('.csv'):
                    file_path = os.path.join('data', file)
                    # For large files, create a sample version
                    if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                        sample_file = f"data/sample_{file}"
                        # Create sample with first 1000 rows
                        import pandas as pd
                        df = pd.read_csv(file_path, nrows=1000)
                        df.to_csv(sample_file, index=False)
                        zipf.write(sample_file)
                        os.remove(sample_file)  # Clean up
                        print(f"   Added: {sample_file} (sample of {file})")
                    else:
                        zipf.write(file_path)
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                        print(f"   Added: {file_path}")
    
    # Get final zip size
    zip_size = os.path.getsize(zip_filename)
    
    print(f"\nCOMPLETE PROJECT ZIP CREATED!")
    print(f"   File name: {zip_filename}")
    print(f"   Zip size: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")
    print(f"   Files included: {file_count}")
    print(f"   Location: {os.path.abspath(zip_filename)}")
    
    print(f"\nCONTENTS:")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file_info in zipf.filelist:
            print(f"   - {file_info.filename}")
    
    print(f"\nPROJECT INCLUDES:")
    print(f"   - All Python source code")
    print(f"   - Dataset samples")
    print(f"   - Model outputs")
    print(f"   - Requirements file")
    print(f"   - Complete project structure")
    
    print(f"\n" + "=" * 60)
    print("ENTIRE PROJECT ZIPPED SUCCESSFULLY!")
    print("=" * 60)
    
    return zip_filename

if __name__ == "__main__":
    zip_entire_project()





