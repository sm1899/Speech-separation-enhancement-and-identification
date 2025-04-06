import os
import argparse
import torch
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
from pesq import pesq
from mir_eval.separation import bss_eval_sources

# Import from src
from src.model_utils import load_wavlm_model, SpeakerEncoder
from src.separation_utils import initialize_sepformer
from src.sepformer_lora import apply_lora_to_sepformer, load_custom_lora_weights
from finetune_combined import Vox2MixDataset, SpeakerDiscriminator, load_model

# Constants
VOX2MIX_DIR = "vox2mix_dataset"
RESULTS_DIR = "combined_model_lora_results"
SEPARATED_DIR = "separated_combined"
BATCH_SIZE = 1  # Process one sample at a time for evaluation
SAMPLE_RATE = 16000
# LoRA parameters (same as in finetune_combined.py)
SEPFORMER_LORA_RANK = 8
SEPFORMER_LORA_ALPHA = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_separation(mixture, source1, source2, estimated_sources):
    """
    Evaluate the quality of separation using SDR, SIR, and SAR.
    Args:
        mixture: Original mixture waveform
        source1: Original source 1 waveform
        source2: Original source 2 waveform
        estimated_sources: Estimated separated sources
    Returns:
        Dictionary containing SDR, SIR, and SAR metrics
    """
    # Convert to numpy for mir_eval
    if torch.is_tensor(source1):
        source1 = source1.cpu().numpy()
    if torch.is_tensor(source2):
        source2 = source2.cpu().numpy()
    if torch.is_tensor(estimated_sources):
        estimated_sources = estimated_sources.cpu().numpy()
    
    # Ensure sources are 2D arrays [num_sources, num_samples]
    references = np.stack([source1, source2], axis=0)
    
    # Calculate metrics for both possible permutations and pick the best
    sdr, sir, sar, perm = bss_eval_sources(references, estimated_sources, compute_permutation=True)
    
    # Return metrics
    return {
        'SDR': sdr.mean(),
        'SIR': sir.mean(),
        'SAR': sar.mean(),
        'permutation': perm
    }

def evaluate_pesq(reference, enhanced, fs=16000):
    """
    Evaluate Perceptual Evaluation of Speech Quality (PESQ).
    Args:
        reference: Reference clean speech
        enhanced: Enhanced/separated speech
        fs: Sampling rate
    Returns:
        PESQ score
    """
    try:
        # Convert to numpy if needed
        if torch.is_tensor(reference):
            reference = reference.cpu().numpy()
        if torch.is_tensor(enhanced):
            enhanced = enhanced.cpu().numpy()
        
        # Ensure audio is mono and the right format
        if len(reference.shape) > 1:
            reference = np.mean(reference, axis=1)
        if len(enhanced.shape) > 1:
            enhanced = np.mean(enhanced, axis=1)
        
        # Normalize audio to [-1, 1]
        reference = reference / np.max(np.abs(reference))
        enhanced = enhanced / np.max(np.abs(enhanced))
        
        # Ensure same length
        min_len = min(len(reference), len(enhanced))
        reference = reference[:min_len]
        enhanced = enhanced[:min_len]
        
        # Calculate PESQ
        score = pesq(fs, reference, enhanced, 'wb')  # Wide-band PESQ
        return score
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return float('nan')

def load_generator_model(model_dir):
    """
    Load the generator model (SepFormer) with LoRA weights
    
    Args:
        model_dir: Directory containing model weights
        
    Returns:
        Loaded model with LoRA adapters
    """
    # Initialize base model
    print("Initializing SepFormer base model...")
    generator = initialize_sepformer(freeze_params=False).to(DEVICE)
    
    # Apply LoRA architecture first (this must happen before loading weights)
    print(f"Applying LoRA architecture with rank={SEPFORMER_LORA_RANK}, alpha={SEPFORMER_LORA_ALPHA}")
    generator = apply_lora_to_sepformer(
        generator,
        lora_rank=SEPFORMER_LORA_RANK,
        lora_alpha=SEPFORMER_LORA_ALPHA
    )
    
    # Look for LoRA weights
    lora_path = os.path.join(model_dir, "final_checkpoint_generator_lora.pt")
    if os.path.exists(lora_path):
        print(f"Loading generator LoRA weights from {lora_path}")
        generator = load_custom_lora_weights(generator, lora_path)
    else:
        print(f"LoRA weights not found at {lora_path}, checking for checkpoint files")
        # Try to find the latest epoch checkpoint with LoRA
        checkpoint_files = [f for f in os.listdir(model_dir) 
                          if f.startswith("checkpoint_epoch_") and f.endswith("_generator_lora.pt")]
        if checkpoint_files:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.split("_")[2]))
            latest_checkpoint = checkpoint_files[-1]
            lora_path = os.path.join(model_dir, latest_checkpoint)
            print(f"Loading generator LoRA weights from latest checkpoint: {lora_path}")
            generator = load_custom_lora_weights(generator, lora_path)
        else:
            print("WARNING: No LoRA weights found. Using initialization weights.")
    
    # Set to evaluation mode
    generator.eval()
    
    # Print parameter stats
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"SepFormer with LoRA - Total parameters: {total_params:,}")
    print(f"SepFormer with LoRA - Trainable parameters: {trainable_params:,}")
    print(f"SepFormer with LoRA - Parameter efficiency: {trainable_params/total_params*100:.2f}%")
    
    return generator

def process_and_save_audio(generator, dataset, output_dir, device):
    """
    Process the test dataset, separate sources, and save the results.
    Args:
        generator: Trained generator model
        dataset: Test dataset
        output_dir: Directory to save results
        device: Device to run the model on
    Returns:
        DataFrame with evaluation metrics
    """
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "separated"), exist_ok=True)
    
    results = []
    
    with torch.no_grad(), tqdm(total=len(dataset)) as pbar:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            
            # Move to device and add batch dimension (only for single samples)
            # SepFormer expects [batch_size, time]
            mixture = sample['mixture'].to(device).unsqueeze(0)  # [1, T]
            source1 = sample['source1'].to(device).unsqueeze(0)  # [1, T]
            source2 = sample['source2'].to(device).unsqueeze(0)  # [1, T]
            
            # Forward pass through generator
            estimated_sources = generator(mixture)  # [1, T, 2]
            
            # Separate the estimated sources
            est_source1 = estimated_sources[:, :, 0]  # [1, T]
            est_source2 = estimated_sources[:, :, 1]  # [1, T]
            
            # Convert to CPU for saving
            mixture_np = mixture.squeeze().cpu().numpy()
            source1_np = source1.squeeze().cpu().numpy()
            source2_np = source2.squeeze().cpu().numpy()
            est_source1_np = est_source1.squeeze().cpu().numpy()
            est_source2_np = est_source2.squeeze().cpu().numpy()
            
            # Save separated sources
            mixture_id = sample['mixture_id']
            sf.write(os.path.join(output_dir, "separated", f"separated_mixture_{mixture_id}_source1.wav"), 
                    est_source1_np, SAMPLE_RATE)
            sf.write(os.path.join(output_dir, "separated", f"separated_mixture_{mixture_id}_source2.wav"), 
                    est_source2_np, SAMPLE_RATE)
            
            # Evaluate separation quality
            estimated_sources_np = np.stack([est_source1_np, est_source2_np], axis=0)
            
            # Evaluate
            separation_metrics = evaluate_separation(
                mixture_np, source1_np, source2_np, estimated_sources_np
            )
            
            # Apply the best permutation
            perm = separation_metrics['permutation']
            est_source1_perm = estimated_sources_np[perm[0]]
            est_source2_perm = estimated_sources_np[perm[1]]
            
            # Calculate PESQ
            pesq_score1 = evaluate_pesq(source1_np, est_source1_perm)
            pesq_score2 = evaluate_pesq(source2_np, est_source2_perm)
            pesq_avg = (pesq_score1 + pesq_score2) / 2
            
            # Store results
            results.append({
                'mixture_id': mixture_id,
                'speaker1_id': sample['speaker1_id'],
                'speaker2_id': sample['speaker2_id'],
                'SDR': separation_metrics['SDR'],
                'SIR': separation_metrics['SIR'],
                'SAR': separation_metrics['SAR'],
                'PESQ_1': pesq_score1,
                'PESQ_2': pesq_score2,
                'PESQ_avg': pesq_avg,
                'separated_source1': os.path.join(output_dir, "separated", f"separated_mixture_{mixture_id}_source1.wav"),
                'separated_source2': os.path.join(output_dir, "separated", f"separated_mixture_{mixture_id}_source2.wav")
            })
            
            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"Processing {mixture_id}")
            pbar.set_postfix({
                'SDR': f"{separation_metrics['SDR']:.2f}",
                'SIR': f"{separation_metrics['SIR']:.2f}",
                'SAR': f"{separation_metrics['SAR']:.2f}",
                'PESQ': f"{pesq_avg:.2f}"
            })
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
    
    # Save metrics summary
    metrics_summary = {
        'SDR': results_df['SDR'].mean(),
        'SIR': results_df['SIR'].mean(),
        'SAR': results_df['SAR'].mean(),
        'PESQ': results_df['PESQ_avg'].mean()
    }
    
    with open(os.path.join(output_dir, "metrics_summary.txt"), 'w') as f:
        for metric, value in metrics_summary.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Plot metrics distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['SDR'], bins=20)
    plt.title(f'SDR Distribution (Mean: {metrics_summary["SDR"]:.2f})')
    plt.xlabel('SDR (dB)')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    plt.hist(results_df['SIR'], bins=20)
    plt.title(f'SIR Distribution (Mean: {metrics_summary["SIR"]:.2f})')
    plt.xlabel('SIR (dB)')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    plt.hist(results_df['SAR'], bins=20)
    plt.title(f'SAR Distribution (Mean: {metrics_summary["SAR"]:.2f})')
    plt.xlabel('SAR (dB)')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 4)
    plt.hist(results_df['PESQ_avg'], bins=20)
    plt.title(f'PESQ Distribution (Mean: {metrics_summary["PESQ"]:.2f})')
    plt.xlabel('PESQ')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_distribution.png"))
    
    return results_df, metrics_summary

def main(args):
    # Load models
    print(f"Loading WavLM model: {args.wavlm_model}")
    wavlm_model, feature_extractor = load_wavlm_model(args.wavlm_model, DEVICE)
    
    # Initialize SepFormer (generator)
    print("Loading generator model...")
    generator = load_generator_model(args.model_dir)
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = Vox2MixDataset(args.data_dir, split="test", sample_rate=SAMPLE_RATE)
    
    # Process and evaluate
    print("Processing test dataset and evaluating...")
    results_df, metrics_summary = process_and_save_audio(
        generator, test_dataset, args.output_dir, DEVICE
    )
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"SDR: {metrics_summary['SDR']:.4f} dB")
    print(f"SIR: {metrics_summary['SIR']:.4f} dB")
    print(f"SAR: {metrics_summary['SAR']:.4f} dB")
    print(f"PESQ: {metrics_summary['PESQ']:.4f}")
    
    print(f"\nDetailed results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate combined model")
    parser.add_argument("--data_dir", type=str, default=VOX2MIX_DIR, help="Directory containing Vox2Mix dataset")
    parser.add_argument("--model_dir", type=str, default=RESULTS_DIR, help="Directory containing trained models")
    parser.add_argument("--output_dir", type=str, default=SEPARATED_DIR, help="Directory to save separated audio and results")
    parser.add_argument("--wavlm_model", type=str, default="microsoft/wavlm-base-plus", help="WavLM model name or path")
    args = parser.parse_args()
    
    main(args) 