#!/usr/bin/env python3
"""
Speech Separation with SepFormer

This script separates mixed utterances using the SepFormer model from SpeechBrain.
"""

import os
import torch
import argparse
from tqdm import tqdm

from src.separation_utils import initialize_sepformer, process_dataset, evaluate_separation

def main():
    # Hardcoded paths for simplicity
    dataset_dir = "vox2mix_dataset"
    output_dir = "separated_vox2mix"
    model_source = "speechbrain/sepformer-wsj02mix"
    sample_rate = 8000  # SepFormer model expects 8kHz input
    
    # Print separation parameters
    print("Running speech separation with SepFormer:")
    print(f"Model Source: {model_source}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Sample Rate: {sample_rate} Hz")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the SepFormer model
    print("Initializing SepFormer model...")
    model = initialize_sepformer(
        model_source=model_source,
        savedir=os.path.join(output_dir, "models"),
        device=None  # Use default (CUDA if available)
    )
    
    # Process the test set
    print("Processing test mixtures...")
    test_results = process_dataset(
        model=model,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        is_test=True,
        sample_rate=sample_rate
    )
    
    # Evaluate separation quality on test set
    print("Evaluating separation quality...")
    test_ground_truth_dir = os.path.join(dataset_dir, "test", "sources")
    test_separated_dir = os.path.join(output_dir, "test", "separated")
    test_metadata_path = os.path.join(dataset_dir, "test", "metadata.csv")
    
    if os.path.exists(test_metadata_path):
        test_metrics = evaluate_separation(
            ground_truth_dir=test_ground_truth_dir,
            separated_dir=test_separated_dir,
            metadata_path=test_metadata_path
        )
        
        print("Test Set Separation Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "separation_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Separation Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    else:
        print("Metadata file not found. Skipping evaluation.")
    
    print("Speech separation completed!")
    print(f"Separated audio saved to {output_dir}")
    print("Now you can use evaluate_enhance.py to identify speakers in the separated audio.")

if __name__ == "__main__":
    main() 