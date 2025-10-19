# ML Challenge 2025 - Smart Product Pricing

A machine learning solution for predicting product prices based on catalog content and product descriptions.

## 🎯 Project Overview

This project implements a price prediction model for the ML Challenge 2025 Smart Product Pricing Challenge. The model analyzes product descriptions and predicts optimal prices using advanced machine learning techniques.

## 📊 Results

- **Model Performance**: 59.51% SMAPE (Symmetric Mean Absolute Percentage Error)
- **Predictions**: 75,000 product price predictions
- **Price Range**: $1.78 - $160.09
- **Average Predicted Price**: $16.86

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-challenge-2025-pricing.git
cd ml-challenge-2025-pricing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the complete pipeline:
```bash
python run_pipeline.py
```

## 📁 Project Structure

```
ml-challenge-2025-pricing/
├── src/                          # Source code
│   ├── 1_eda.py                 # Exploratory data analysis
│   ├── 2_baseline_clean.py      # Main ML model
│   ├── 3_text_embeddings.py     # Text feature extraction
│   ├── 4_image_embeddings.py    # Image feature extraction
│   ├── 5_train_models.py        # Advanced model training
│   ├── 6_stack_and_blend.py     # Model ensemble
│   └── _0_utils.py              # Utility functions
├── data/                         # Dataset
│   ├── train.csv                # Training data (75k samples)
│   ├── test.csv                 # Test data (75k samples)
│   └── images_cache/            # Product images
├── outputs/                      # Model outputs
│   ├── sub_baseline_clean.csv   # Main predictions
│   └── oof_baseline_clean.csv   # Cross-validation results
├── models/                       # Trained models
│   └── lgb_baseline_clean.pkl   # LightGBM model
├── run_pipeline.py              # Main pipeline runner
├── prepare_submission.py        # Competition submission prep
├── validate_output.py           # Output validation
├── view_results.py              # Results visualization
└── README.md                    # This file
```

## 🔧 Features

### Data Processing
- **Text Analysis**: TF-IDF vectorization with 10,000 features
- **Feature Engineering**: Length, word count, digits, IPQ extraction
- **Image Processing**: Ready for visual feature extraction

### Machine Learning
- **Algorithm**: LightGBM (Gradient Boosting)
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Target Transformation**: Log-transformed for better performance
- **Regularization**: Early stopping and parameter tuning

### Model Performance
- **SMAPE Score**: 59.51% (baseline performance)
- **Training Time**: ~3 minutes (10% sample)
- **Prediction Time**: <1 second for 75k samples

## 📈 Usage

### Run Complete Pipeline
```bash
python run_pipeline.py
```

### View Results
```bash
python view_results.py
```

### Validate Output
```bash
python validate_output.py
```

### Prepare Competition Submission
```bash
python prepare_submission.py
```

## 🎯 Competition Compliance

- ✅ **Format**: Perfect CSV with sample_id and price columns
- ✅ **Samples**: All 75,000 predictions present
- ✅ **Data Types**: Correct (int64 for sample_id, float64 for price)
- ✅ **No External Data**: Uses only provided dataset
- ✅ **Fair Play**: No external price lookup

## 📊 Model Architecture

### Feature Engineering
1. **Text Features**: TF-IDF with n-grams (1-2)
2. **Numerical Features**: Text length, word count, digit count
3. **Derived Features**: Item Pack Quantity (IPQ) extraction
4. **Image Features**: Ready for CLIP embeddings

### Model Training
- **Algorithm**: LightGBM
- **Objective**: Regression L1 (MAE)
- **Cross-Validation**: 3-fold
- **Early Stopping**: 20 rounds
- **Learning Rate**: 0.1

## 🔮 Future Improvements

- [ ] Use full dataset (100% instead of 10%)
- [ ] Add image features (CLIP embeddings)
- [ ] Implement ensemble methods
- [ ] Try advanced text embeddings (BERT, RoBERTa)
- [ ] Hyperparameter optimization
- [ ] Feature selection and engineering

## 📝 Methodology

1. **Data Analysis**: Exploratory analysis of product descriptions and prices
2. **Feature Engineering**: Text processing and numerical feature extraction
3. **Model Training**: LightGBM with cross-validation
4. **Prediction**: Generate 75,000 price predictions
5. **Validation**: Ensure competition format compliance

## 🏆 Competition Results

- **Submission File**: `test_out.csv`
- **Format**: CSV with sample_id and price columns
- **Validation**: All requirements met
- **Status**: Ready for submission

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Paskal Garry Vinu F - ML Challenge 2025 Participant

## 🙏 Acknowledgments

- ML Challenge 2025 organizers
- LightGBM developers
- Scikit-learn community
- Open source contributors

---

**Note**: This project is for educational and competition purposes. All data used is from the provided ML Challenge 2025 dataset.
