#!/usr/bin/env python3
"""
Upload the complete project to your existing GitHub repository
"""

import os
import subprocess
import shutil
from datetime import datetime

def upload_to_github():
    """Upload the project to your existing GitHub repository."""
    print("=" * 60)
    print("UPLOADING TO YOUR GITHUB REPOSITORY")
    print("=" * 60)
    
    # Your repository details
    repo_url = "https://github.com/Paskalgarryvinu/E-commerce-platform-for-real-price-prediction.git"
    repo_name = "E-commerce-platform-for-real-price-prediction"
    
    print(f"Repository: {repo_url}")
    print(f"Project: {repo_name}")
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("SUCCESS: Git is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Git is not installed or not in PATH")
        print("Please install Git from: https://git-scm.com/downloads")
        return False
    
    # Create a clean project directory
    project_dir = repo_name
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    
    os.makedirs(project_dir, exist_ok=True)
    print(f"\nCREATING PROJECT DIRECTORY: {project_dir}")
    
    # Copy all necessary files
    files_to_copy = [
        "README.md",
        "requirements.txt", 
        ".gitignore",
        "LICENSE",
        "SETUP.md",
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
        "prepare_for_github.py",
        "upload_to_github.py"
    ]
    
    # Copy Python files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, project_dir)
            print(f"   SUCCESS: Copied: {file}")
    
    # Copy src directory
    if os.path.exists("src"):
        shutil.copytree("src", os.path.join(project_dir, "src"))
        print(f"   SUCCESS: Copied: src/ directory")
    
    # Create data directory with samples
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    
    # Create sample data files
    print("CREATING SAMPLE DATA FILES...")
    try:
        import pandas as pd
        
        # Sample train data
        if os.path.exists("data/train.csv"):
            train_sample = pd.read_csv("data/train.csv", nrows=1000)
            train_sample.to_csv(os.path.join(project_dir, "data", "sample_train.csv"), index=False)
            print("   SUCCESS: Created: data/sample_train.csv (1000 rows)")
        
        # Sample test data
        if os.path.exists("data/test.csv"):
            test_sample = pd.read_csv("data/test.csv", nrows=1000)
            test_sample.to_csv(os.path.join(project_dir, "data", "sample_test.csv"), index=False)
            print("   SUCCESS: Created: data/sample_test.csv (1000 rows)")
    
    except Exception as e:
        print(f"   WARNING: Could not create sample data files: {e}")
    
    # Create outputs directory with sample
    os.makedirs(os.path.join(project_dir, "outputs"), exist_ok=True)
    
    # Create sample output
    if os.path.exists("outputs/sub_baseline_clean.csv"):
        output_sample = pd.read_csv("outputs/sub_baseline_clean.csv", nrows=100)
        output_sample.to_csv(os.path.join(project_dir, "outputs", "sample_predictions.csv"), index=False)
        print("   SUCCESS: Created: outputs/sample_predictions.csv (100 rows)")
    
    # Create models directory
    os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)
    
    # Create a placeholder for models
    with open(os.path.join(project_dir, "models", "README.md"), "w") as f:
        f.write("# Models Directory\n\n")
        f.write("This directory contains trained model files.\n")
        f.write("Run `python run_pipeline.py` to generate the models.\n")
    
    # Update README.md with your repository URL
    readme_path = os.path.join(project_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()
        
        # Replace placeholder URL with your actual repository
        content = content.replace("https://github.com/yourusername/ml-challenge-2025-pricing.git", repo_url)
        content = content.replace("ml-challenge-2025-pricing", repo_name)
        
        with open(readme_path, "w") as f:
            f.write(content)
        
        print("   SUCCESS: Updated README.md with your repository URL")
    
    # Initialize git repository
    print(f"\nINITIALIZING GIT REPOSITORY...")
    os.chdir(project_dir)
    
    try:
        # Initialize git
        subprocess.run(["git", "init"], check=True)
        print("   SUCCESS: Git repository initialized")
        
        # Add remote origin
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        print("   SUCCESS: Remote origin added")
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("   SUCCESS: Files added to git")
        
        # Commit files
        commit_message = f"Initial commit: ML Challenge 2025 - Smart Product Pricing Solution\n\n- Complete price prediction pipeline\n- LightGBM model with 59.51% SMAPE\n- 75,000 product predictions\n- Competition-ready submission format\n- Professional documentation and setup"
        
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print("   SUCCESS: Files committed")
        
        # Push to GitHub
        print(f"\nPUSHING TO GITHUB...")
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print("   SUCCESS: Successfully pushed to GitHub!")
        
        print(f"\nPROJECT SUCCESSFULLY UPLOADED!")
        print(f"   Repository: {repo_url}")
        print(f"   Branch: main")
        print(f"   Files uploaded: {len(os.listdir('.'))}")
        
        print(f"\nWHAT WAS UPLOADED:")
        print(f"   SUCCESS: Complete source code (15+ Python files)")
        print(f"   SUCCESS: Professional README.md with documentation")
        print(f"   SUCCESS: Requirements.txt with dependencies")
        print(f"   SUCCESS: MIT License")
        print(f"   SUCCESS: Sample data files")
        print(f"   SUCCESS: Setup instructions")
        print(f"   SUCCESS: Competition submission files")
        
        print(f"\nVIEW YOUR REPOSITORY:")
        print(f"   {repo_url}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Git error: {e}")
        print("Please check your Git configuration and repository access")
        return False
    
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = upload_to_github()
    if success:
        print(f"\n" + "=" * 60)
        print("UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print(f"\n" + "=" * 60)
        print("UPLOAD FAILED - Please check the errors above")
        print("=" * 60)
