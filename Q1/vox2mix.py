#!/usr/bin/env python3
"""
VoxCeleb2 Mixture Dataset Creator

This script creates a dataset of mixed utterances from the VoxCeleb2 dataset
for multi-speaker scenario evaluation.
"""

import os
import argparse
import torch
from tqdm import tqdm

from src.mixing_utils import create_mixture_dataset

def main():
    # Hardcoded paths for simplicity
    vox2_path = "Speech2025Datasets/vox2"
    output_dir = "vox2mix_dataset"
    first_n_train_speakers = 50
    first_n_test_speakers = 50
    
    # Print dataset parameters
    print("Creating VoxCeleb2 mixture dataset with the following parameters:")
    print(f"VoxCeleb2 Path: {vox2_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Number of Train Speakers: {first_n_train_speakers}")
    print(f"Number of Test Speakers: {first_n_test_speakers}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all speaker directories in sorted order
    all_speaker_dirs = sorted([d for d in os.listdir(vox2_path) if os.path.isdir(os.path.join(vox2_path, d))])
    
    # First 50 speakers for training
    train_speakers = all_speaker_dirs[:first_n_train_speakers]
    
    # Next 50 speakers for testing
    test_speakers = all_speaker_dirs[first_n_train_speakers:first_n_train_speakers + first_n_test_speakers]
    
    print(f"Selected {len(train_speakers)} speakers for training and {len(test_speakers)} speakers for testing")
    
    # Create mixture dataset
    create_mixture_dataset(
        train_speakers=train_speakers,
        test_speakers=test_speakers,
        vox2_path=vox2_path,
        output_dir=output_dir,
        num_train_mixtures=500,  # Reduce number for faster execution
        num_test_mixtures=200,   # Reduce number for faster execution
        snr_range=(-5, 5),
        overlap_ratio_range=(0.5, 1.0),
        sample_rate=16000
    )
    
    print("VoxCeleb2 mixture dataset creation completed!")
    print(f"Dataset saved to {output_dir}")
    print("Now you can use separation.py to separate the mixtures.")

if __name__ == "__main__":
    main() 