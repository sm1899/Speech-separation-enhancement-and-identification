#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from src.data_utils import VoxCelebDataset, create_verification_trials
from src.model_utils import load_wavlm_model, extract_embeddings
from src.evaluation import run_speaker_verification, plot_roc_curve

# Paths and constants
VOX1_DIR = "/home/m23mac008/assignment_2_m23mac008/q1/Speech2025Datasets/vox1"
MODEL_NAME = "microsoft/wavlm-base-plus"
BATCH_SIZE = 8
NUM_TRIALS = 20000

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print("Loading VoxCeleb1 dataset...")
    vox1_dataset = VoxCelebDataset(VOX1_DIR)
    
    # Use all samples
    print(f"Using all available samples: {len(vox1_dataset)}")
    
    vox1_loader = DataLoader(vox1_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded dataset with {len(vox1_dataset)} samples from {len(vox1_dataset.speaker_dirs)} speakers")
    
    # Load pre-trained WavLM model
    model, feature_extractor = load_wavlm_model(MODEL_NAME, device)
    
    # Create verification trials
    print(f"Creating {NUM_TRIALS} verification trials...")
    verification_trials = create_verification_trials(vox1_dataset, num_trials=NUM_TRIALS)
    
    # Run speaker verification
    print("Running speaker verification with pre-trained WavLM model...")
    metrics, scores, labels = run_speaker_verification(
        model, feature_extractor, vox1_loader, verification_trials, device
    )
    
    # Print results
    print("\nPre-trained WavLM Base Plus Results on VoxCeleb1:")
    print(f"EER: {metrics['EER']:.2f}%")
    print(f"TAR@1%FAR: {metrics['TAR@1%FAR']:.2f}%")
    print(f"Accuracy: {metrics['Accuracy']:.2f}%")
    
    # Save results to file
    os.makedirs("results", exist_ok=True)
    with open("results/pretrained_results.txt", "w") as f:
        f.write(f"Pre-trained WavLM Base Plus Results on VoxCeleb1:\n")
        f.write(f"EER: {metrics['EER']:.2f}%\n")
        f.write(f"TAR@1%FAR: {metrics['TAR@1%FAR']:.2f}%\n")
        f.write(f"Accuracy: {metrics['Accuracy']:.2f}%\n")
    
    # Save scores and labels for future comparison
    np.savez("results/pretrained_scores.npz", scores=scores, labels=labels)
    
    # Plot ROC curve
    plt = plot_roc_curve(scores, labels, title="ROC Curve - Pre-trained WavLM on VoxCeleb1")
    plt.savefig("results/pretrained_roc.png")
    plt.close()
    
    print("Evaluation complete. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main() 