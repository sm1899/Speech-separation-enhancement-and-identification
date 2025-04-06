#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random

from src.data_utils import VoxCelebDataset, create_verification_trials
from src.model_utils import load_wavlm_model, extract_embeddings, create_lora_model
from src.evaluation import run_speaker_verification, plot_roc_curve
from finetune_lora import SpeakerVerificationModel

# Paths and constants
VOX1_DIR = "/home/m23mac008/assignment_2_m23mac008/q1/Speech2025Datasets/vox1"
MODEL_PATH = "models/best_model.pth"  # Path to the fine-tuned model weights
MODEL_NAME = "microsoft/wavlm-base-plus"
BATCH_SIZE = 8
NUM_TRIALS = 20000  # Increased for more robust evaluation
EMBEDDING_DIM = 256
NUM_TRAIN_SPEAKERS = 100  # Same as in training

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
    
    # Load pre-trained WavLM model for feature extraction
    _, feature_extractor = load_wavlm_model(MODEL_NAME, device)
    
    # For evaluation, we don't need LoRA. We'll use the original model and custom loading
    # Load the base WavLM model
    base_model, _ = load_wavlm_model(MODEL_NAME, device)
    
    # Create a fresh model without LoRA
    model = SpeakerVerificationModel(base_model, NUM_TRAIN_SPEAKERS, EMBEDDING_DIM).to(device)
    
    # Load fine-tuned weights with remapping
    print(f"Loading fine-tuned model weights from {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Create new state dict with only the projection and arcface parts
        new_state_dict = {}
        
        # Extract just the projection and arcface weights (not the WavLM model weights)
        projection_keys = ['projection.weight', 'projection.bias', 'arc_face.weight']
        for key in projection_keys:
            if key in state_dict:
                new_state_dict[key] = state_dict[key]
                print(f"Loaded {key} from fine-tuned model")
        
        # Load our filtered state dict
        model.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded fine-tuned model weights for projection and ArcFace layers")
    else:
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    
    model.eval()
    
    # Create verification trials
    print(f"Creating {NUM_TRIALS} verification trials...")
    verification_trials = create_verification_trials(vox1_dataset, num_trials=NUM_TRIALS)
    
    # Run speaker verification
    print("Running speaker verification with fine-tuned model...")
    metrics, scores, labels = run_speaker_verification(
        model, feature_extractor, vox1_loader, verification_trials, device
    )
    
    # Print results
    print("\nFine-tuned WavLM Results on VoxCeleb1:")
    print(f"EER: {metrics['EER']:.2f}%")
    print(f"TAR@1%FAR: {metrics['TAR@1%FAR']:.2f}%")
    print(f"Accuracy: {metrics['Accuracy']:.2f}%")
    
    # Load pre-trained results for comparison if available
    if os.path.exists("results/pretrained_results.txt"):
        with open("results/pretrained_results.txt", "r") as f:
            pretrained_results = f.readlines()
        
        print("\nComparison with Pre-trained Model:")
        print(f"Pre-trained - {pretrained_results[1].strip()}")
        print(f"Fine-tuned - EER: {metrics['EER']:.2f}%")
        
        print(f"Pre-trained - {pretrained_results[2].strip()}")
        print(f"Fine-tuned - TAR@1%FAR: {metrics['TAR@1%FAR']:.2f}%")
        
        print(f"Pre-trained - {pretrained_results[3].strip()}")
        print(f"Fine-tuned - Accuracy: {metrics['Accuracy']:.2f}%")
    
    # Save results to file
    os.makedirs("results", exist_ok=True)
    with open("results/finetuned_results.txt", "w") as f:
        f.write(f"Fine-tuned WavLM Results on VoxCeleb1:\n")
        f.write(f"EER: {metrics['EER']:.2f}%\n")
        f.write(f"TAR@1%FAR: {metrics['TAR@1%FAR']:.2f}%\n")
        f.write(f"Accuracy: {metrics['Accuracy']:.2f}%\n")
    
    # Plot ROC curve
    plt = plot_roc_curve(scores, labels, title="ROC Curve - Fine-tuned WavLM on VoxCeleb1")
    plt.savefig("results/finetuned_roc.png")
    
    # Compare ROC curves if pre-trained scores are available
    if os.path.exists("results/pretrained_roc.png"):
        try:
            # Load pre-trained scores and labels if saved
            pretrained_results_path = "results/pretrained_scores.npz"
            if os.path.exists(pretrained_results_path):
                pretrained_data = np.load(pretrained_results_path)
                pretrained_scores = pretrained_data['scores']
                pretrained_labels = pretrained_data['labels']
                
                # Plot comparison
                plt.figure(figsize=(10, 8))
                fpr_ft, tpr_ft, _ = roc_curve(labels, scores)
                roc_auc_ft = auc(fpr_ft, tpr_ft)
                
                fpr_pt, tpr_pt, _ = roc_curve(pretrained_labels, pretrained_scores)
                roc_auc_pt = auc(fpr_pt, tpr_pt)
                
                plt.plot(fpr_ft, tpr_ft, lw=2, label=f'Fine-tuned (AUC = {roc_auc_ft:.3f})')
                plt.plot(fpr_pt, tpr_pt, lw=2, linestyle='--', label=f'Pre-trained (AUC = {roc_auc_pt:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve Comparison')
                plt.legend(loc="lower right")
                plt.grid()
                plt.savefig("results/roc_comparison.png")
                plt.close()
        except Exception as e:
            print(f"Could not create comparison plot: {e}")
    
    # Save scores and labels for future comparison
    np.savez("results/finetuned_scores.npz", scores=scores, labels=labels)
    
    print("Evaluation complete. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main() 