#!/usr/bin/env python3
"""
Prepare the project for GitHub upload
"""

import os
import shutil
import zipfile
from datetime import datetime

def prepare_for_github():
    """Prepare the project for GitHub upload."""
    print("=" * 60)
    print("PREPARING PROJECT FOR GITHUB")
    print("=" * 60)
    
    # Create a clean project directory for GitHub
    github_dir = "ml-challenge-2025-pricing"
    
    if os.path.exists(github_dir):
        shutil.rmtree(github_dir)
    
    os.makedirs(github_dir, exist_ok=True)
    
    print(f"Creating clean project directory: {github_dir}")
    
    # Files to copy (excluding large data files)
    files_to_copy = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "run_pipeline.py",
        "prepare_submission.py",
        "validate_output.py",
        "view_results.py",
        "see_my_output.py",
        "summary.py",
        "create_submission.py",
        "improve_model.py",
        "run_full_dataset.py",
        "compare_results.py",
        "zip_entire_project.py",
        "prepare_for_github.py"
    ]
    
    # Copy Python files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, github_dir)
            print(f"   Copied: {file}")
    
    # Copy src directory
    if os.path.exists("src"):
        shutil.copytree("src", os.path.join(github_dir, "src"))
        print(f"   Copied: src/ directory")
    
    # Create sample data directory
    os.makedirs(os.path.join(github_dir, "data"), exist_ok=True)
    
    # Create sample data files (first 1000 rows)
    print("Creating sample data files...")
    try:
        import pandas as pd
        
        # Sample train data
        if os.path.exists("data/train.csv"):
            train_sample = pd.read_csv("data/train.csv", nrows=1000)
            train_sample.to_csv(os.path.join(github_dir, "data", "sample_train.csv"), index=False)
            print("   Created: data/sample_train.csv (1000 rows)")
        
        # Sample test data
        if os.path.exists("data/test.csv"):
            test_sample = pd.read_csv("data/test.csv", nrows=1000)
            test_sample.to_csv(os.path.join(github_dir, "data", "sample_test.csv"), index=False)
            print("   Created: data/sample_test.csv (1000 rows)")
    
    except Exception as e:
        print(f"   Warning: Could not create sample data files: {e}")
    
    # Create outputs directory with sample
    os.makedirs(os.path.join(github_dir, "outputs"), exist_ok=True)
    
    # Create sample output
    if os.path.exists("outputs/sub_baseline_clean.csv"):
        output_sample = pd.read_csv("outputs/sub_baseline_clean.csv", nrows=100)
        output_sample.to_csv(os.path.join(github_dir, "outputs", "sample_predictions.csv"), index=False)
        print("   Created: outputs/sample_predictions.csv (100 rows)")
    
    # Create models directory
    os.makedirs(os.path.join(github_dir, "models"), exist_ok=True)
    
    # Create a placeholder for models
    with open(os.path.join(github_dir, "models", "README.md"), "w") as f:
        f.write("# Models Directory\n\n")
        f.write("This directory contains trained model files.\n")
        f.write("Run `python run_pipeline.py` to generate the models.\n")
    
    # Create LICENSE file
    with open(os.path.join(github_dir, "LICENSE"), "w") as f:
        f.write("MIT License\n\n")
        f.write("Copyright (c) 2025 ML Challenge Participant\n\n")
        f.write("Permission is hereby granted, free of charge, to any person obtaining a copy\n")
        f.write("of this software and associated documentation files (the \"Software\"), to deal\n")
        f.write("in the Software without restriction, including without limitation the rights\n")
        f.write("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n")
        f.write("copies of the Software, and to permit persons to whom the Software is\n")
        f.write("furnished to do so, subject to the following conditions:\n\n")
        f.write("The above copyright notice and this permission notice shall be included in all\n")
        f.write("copies or substantial portions of the Software.\n\n")
        f.write("THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n")
        f.write("IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n")
        f.write("FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n")
        f.write("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n")
        f.write("LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n")
        f.write("OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n")
        f.write("SOFTWARE.\n")
    
    # Create setup instructions
    with open(os.path.join(github_dir, "SETUP.md"), "w") as f:
        f.write("# Setup Instructions\n\n")
        f.write("## Quick Start\n\n")
        f.write("1. Clone this repository:\n")
        f.write("```bash\n")
        f.write("git clone https://github.com/yourusername/ml-challenge-2025-pricing.git\n")
        f.write("cd ml-challenge-2025-pricing\n")
        f.write("```\n\n")
        f.write("2. Install dependencies:\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        f.write("3. Run the pipeline:\n")
        f.write("```bash\n")
        f.write("python run_pipeline.py\n")
        f.write("```\n\n")
        f.write("## Data Requirements\n\n")
        f.write("Place your dataset files in the `data/` directory:\n")
        f.write("- `train.csv` - Training data with prices\n")
        f.write("- `test.csv` - Test data for predictions\n")
        f.write("- `images_cache/` - Product images (optional)\n")
    
    # Create zip file for easy upload
    zip_filename = f"ml-challenge-2025-pricing-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    print(f"\nCreating zip file for GitHub upload...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(github_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, github_dir)
                zipf.write(file_path, arcname)
    
    zip_size = os.path.getsize(zip_filename)
    
    print(f"\nGITHUB PROJECT READY!")
    print(f"   Project directory: {github_dir}")
    print(f"   Zip file: {zip_filename}")
    print(f"   Zip size: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")
    
    print(f"\nFILES INCLUDED:")
    for root, dirs, files in os.walk(github_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, github_dir)
            print(f"   - {rel_path}")
    
    print(f"\nNEXT STEPS:")
    print(f"1. Create a new repository on GitHub")
    print(f"2. Upload the files from: {github_dir}")
    print(f"3. Or upload the zip file: {zip_filename}")
    print(f"4. Update the repository URL in README.md")
    
    print(f"\n" + "=" * 60)
    print("PROJECT READY FOR GITHUB!")
    print("=" * 60)
    
    return github_dir, zip_filename

if __name__ == "__main__":
    prepare_for_github()
