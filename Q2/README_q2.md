# MFCC Analysis of Indian Languages

This project performs MFCC feature extraction, visualization, and classification on audio samples from 10 Indian languages.

## Project Structure

- `mfcc_analysis.py` - Main script for feature extraction, visualization, and classification
- `MFCC_Analysis_Report.md` - Comprehensive report of findings and analysis
- `requirements.txt` - Required dependencies
- `mfcc_visualizations/` - Directory containing generated visualizations (created on script execution)
- `confusion_matrix.png` - Classification model evaluation (created on script execution)

## Setup

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Analysis

Execute the main script to perform both MFCC extraction/visualization and language classification:

```bash
python mfcc_analysis.py
```

The script performs the following tasks:
1. Extracts MFCC features from audio samples in the dataset
2. Generates and saves visualizations for comparing languages
3. Builds a Random Forest classifier to predict languages
4. Evaluates the classifier and creates a confusion matrix

## Dataset

The dataset contains audio samples from 10 Indian languages:
- Bengali
- Gujarati
- Hindi
- Kannada
- Malayalam
- Marathi
- Punjabi
- Tamil
- Telugu
- Urdu

For detailed visualization and analysis, we focus on Hindi, Tamil, and Malayalam.

## Results

The analysis results include:
- MFCC visualizations for each language
- Statistical analysis of MFCC coefficients
- Comparison of mean MFCC values across languages
- Classification model performance metrics
- Confusion matrix showing classification accuracy

A detailed interpretation of these results can be found in `MFCC_Analysis_Report.md`. 