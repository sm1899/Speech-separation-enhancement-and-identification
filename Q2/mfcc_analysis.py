import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Set the paths
BASE_DIR = "/home/m23mac008/assignment_2_m23mac008/Language Detection Dataset"
LANGUAGES = ['Bengali', 'Gujarati', 'Hindi', 'Kannada', 'Malayalam', 
            'Marathi', 'Punjabi', 'Tamil', 'Telugu', 'Urdu']

# Select 3 languages for visualization comparison
LANGUAGES_TO_VISUALIZE = ['Hindi', 'Tamil', 'Malayalam']

def extract_mfcc_features(file_path, n_mfcc=13, duration=5):
    """
    Extract MFCC features from an audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    n_mfcc : int, optional
        Number of MFCC coefficients to extract
    duration : int, optional
        Duration in seconds to analyze (from start of file)
        
    Returns:
    --------
    mfcc_features : ndarray
        MFCC features
    sr : int
        Sample rate
    """
    try:
        # Load audio file with specified duration
        y, sr = librosa.load(file_path, sr=16000, duration=duration)
        
        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr)
        
        return mfcc_features, sr
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def visualize_mfcc(mfcc_features, sr, title):
    """
    Visualize MFCC features as a spectrogram.
    
    Parameters:
    -----------
    mfcc_features : ndarray
        MFCC features
    sr : int
        Sample rate
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_features, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()

def extract_features_for_classification(languages=LANGUAGES, n_samples_per_language=100, n_mfcc=13):
    """
    Extract MFCC features from audio files for classification.
    
    Parameters:
    -----------
    languages : list
        List of languages to process
    n_samples_per_language : int
        Number of samples to process per language
    n_mfcc : int
        Number of MFCC coefficients to extract
        
    Returns:
    --------
    X : ndarray
        Features for classification
    y : ndarray
        Labels for classification
    """
    features = []
    labels = []
    
    for lang_idx, language in enumerate(languages):
        lang_dir = os.path.join(BASE_DIR, language)
        audio_files = glob(os.path.join(lang_dir, "*.mp3"))
        
        # Limit the number of samples per language
        audio_files = audio_files[:n_samples_per_language]
        
        print(f"Processing {language} files...")
        for file_path in tqdm(audio_files):
            mfcc_features, _ = extract_mfcc_features(file_path)
            
            if mfcc_features is not None:
                # Compute statistical features from MFCCs
                mfcc_mean = np.mean(mfcc_features, axis=1)
                mfcc_var = np.var(mfcc_features, axis=1)
                
                # Combine mean and variance as features
                feature_vector = np.concatenate([mfcc_mean, mfcc_var])
                
                features.append(feature_vector)
                labels.append(lang_idx)
    
    return np.array(features), np.array(labels)

def perform_classification(X, y):
    """
    Train and evaluate a classifier on MFCC features.
    
    Parameters:
    -----------
    X : ndarray
        Features for classification
    y : ndarray
        Labels for classification
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained classifier
    accuracy : float
        Accuracy of the model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LANGUAGES))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LANGUAGES, yticklabels=LANGUAGES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return model, accuracy

def main():
    # Task A: Extract and visualize MFCC for comparative analysis
    print("Task A: Extracting and visualizing MFCC features")
    
    # Create a directory for saving visualizations
    os.makedirs('mfcc_visualizations', exist_ok=True)
    
    # For each language to visualize, process a few samples
    for language in LANGUAGES_TO_VISUALIZE:
        lang_dir = os.path.join(BASE_DIR, language)
        audio_files = glob(os.path.join(lang_dir, "*.mp3"))
        
        # Process the first 5 samples from each language
        for i, file_path in enumerate(audio_files[:5]):
            mfcc_features, sr = extract_mfcc_features(file_path)
            
            if mfcc_features is not None:
                # Visualize MFCC
                plt.figure(figsize=(12, 5))
                
                # Plot MFCC features
                plt.subplot(1, 1, 1)
                img = librosa.display.specshow(mfcc_features, x_axis='time', sr=sr)
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'MFCC - {language} (Sample {i+1})')
                
                # Save the visualization
                plt.tight_layout()
                plt.savefig(f'mfcc_visualizations/{language}_sample_{i+1}.png')
                plt.close()
                
                # For the first sample of each language, compute and save statistics
                if i == 0:
                    mfcc_mean = np.mean(mfcc_features, axis=1)
                    mfcc_var = np.var(mfcc_features, axis=1)
                    
                    with open(f'mfcc_visualizations/{language}_statistics.txt', 'w') as f:
                        f.write(f"MFCC Statistics for {language}\n")
                        f.write("========================\n\n")
                        f.write("Mean values of MFCC coefficients:\n")
                        for j, mean_val in enumerate(mfcc_mean):
                            f.write(f"Coefficient {j+1}: {mean_val}\n")
                        f.write("\nVariance of MFCC coefficients:\n")
                        for j, var_val in enumerate(mfcc_var):
                            f.write(f"Coefficient {j+1}: {var_val}\n")
    
    # Compare mean MFCC coefficients across languages
    plt.figure(figsize=(12, 8))
    for language in LANGUAGES_TO_VISUALIZE:
        # Load a representative sample
        lang_dir = os.path.join(BASE_DIR, language)
        file_path = glob(os.path.join(lang_dir, "*.mp3"))[0]
        mfcc_features, _ = extract_mfcc_features(file_path)
        
        if mfcc_features is not None:
            mfcc_mean = np.mean(mfcc_features, axis=1)
            plt.plot(mfcc_mean, label=language)
    
    plt.title('Comparison of Mean MFCC Coefficients Across Languages')
    plt.xlabel('MFCC Coefficient')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('mfcc_visualizations/mean_mfcc_comparison.png')
    plt.close()
    
    # Task B: Classification
    print("\nTask B: Building a classifier using MFCC features")
    X, y = extract_features_for_classification()
    model, accuracy = perform_classification(X, y)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 