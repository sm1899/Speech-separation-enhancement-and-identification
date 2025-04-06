import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import soundfile as sf
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
from pathlib import Path
import torchaudio

# Import from src
from src.model_utils import load_wavlm_model, create_lora_model, SpeakerEncoder, SpeakerDiscriminator
from src.separation_utils import initialize_sepformer
from src.data_utils import VoxCelebDataset
from src.metrics import compute_embeddings_for_separated_sources
from src.loss_utils import SI_SDRLoss, PerceptualLoss, ModelBasedPerceptualLoss
from src.sepformer_lora import setup_sepformer_lora_for_pipeline, save_custom_lora_weights

# Constants
VOX2MIX_DIR = "vox2mix_dataset"
RESULTS_DIR = "combined_model_lora_results"
BATCH_SIZE = 1  # SepFormer works best with batch size 1 due to its architecture
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5
LEARNING_RATE_D = 1e-3
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000
# LoRA parameters (default settings for memory efficiency)
LORA_RANK = 16
LORA_ALPHA = 32
SEPFORMER_LORA_RANK = 32
SEPFORMER_LORA_ALPHA = 64
LOSS_WEIGHTS = {
    'separation': 1.0,
    'perceptual': 0.1,
    'adversarial': 0.01
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define device configuration (multi-GPU setup)
GENERATOR_DEVICE = torch.device('cuda:0')  # A6000 (48GB)
DISCRIMINATOR_DEVICE = torch.device('cuda:1')  # A5000 (24GB)
print(f"Using generator device: {GENERATOR_DEVICE}")
print(f"Using discriminator device: {DISCRIMINATOR_DEVICE}")

def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Vox2MixDataset(torch.utils.data.Dataset):
    """Dataset for loading mixed utterances and their sources."""
    def __init__(self, data_dir, split="train", sample_rate=16000, max_duration=8.0):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the Vox2Mix dataset
            split: Either 'train' or 'test'
            sample_rate: Sample rate of the audio files
            max_duration: Maximum audio duration in seconds (to prevent OOM errors)
        """
        self.data_dir = data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        print(f"Dataset will limit audio to {max_duration}s ({self.max_samples} samples)")
        
        # Load metadata
        self.metadata_path = os.path.join(data_dir, split, "metadata.csv")
        self.metadata = np.loadtxt(self.metadata_path, delimiter=',', dtype=str, skiprows=1)
        
        # Filter outliers by checking file sizes (optional)
        # self._filter_large_files()
        
        # Get mixture and source paths
        self.mixture_dir = os.path.join(data_dir, split, "mixtures")
        self.source_dir = os.path.join(data_dir, split, "sources")
        
        print(f"Loaded {len(self.metadata)} {split} samples")
    
    def _filter_large_files(self):
        """Filter out extremely large audio files to prevent OOM errors."""
        filtered_metadata = []
        skipped_count = 0
        
        for item in self.metadata:
            mixture_id = item[0]
            mixture_path = os.path.join(self.mixture_dir, f"mixture_{mixture_id}.wav")
            
            # Skip if file doesn't exist
            if not os.path.exists(mixture_path):
                skipped_count += 1
                continue
                
            # Check file size (in bytes)
            file_size = os.path.getsize(mixture_path)
            
            # Skip files larger than ~10MB (very conservative)
            if file_size > 10 * 1024 * 1024:
                print(f"Skipping large file: {mixture_path} ({file_size/1024/1024:.2f} MB)")
                skipped_count += 1
                continue
                
            filtered_metadata.append(item)
        
        print(f"Filtered out {skipped_count} files from {self.split} set")
        self.metadata = np.array(filtered_metadata)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a mixture and its corresponding sources."""
        # Get mixture ID and paths
        mixture_id = self.metadata[idx, 0]
        
        # Paths for mixture and sources
        mixture_path = os.path.join(self.mixture_dir, f"mixture_{mixture_id}.wav")
        source1_path = os.path.join(self.source_dir, f"source1_{mixture_id}.wav")
        source2_path = os.path.join(self.source_dir, f"source2_{mixture_id}.wav")
        
        try:
            # Load audio files
            mixture, sr = torchaudio.load(mixture_path)
            source1, _ = torchaudio.load(source1_path)
            source2, _ = torchaudio.load(source2_path)
            
            # Check size and print warning for large files
            num_samples = mixture.shape[-1]
            if num_samples > self.max_samples:
                print(f"Warning: Long audio found ({num_samples/sr:.2f}s), ID: {mixture_id}")
            
            # Ensure mono
            mixture = mixture.mean(dim=0) if mixture.shape[0] > 1 else mixture.squeeze(0)
            source1 = source1.mean(dim=0) if source1.shape[0] > 1 else source1.squeeze(0)
            source2 = source2.mean(dim=0) if source2.shape[0] > 1 else source2.squeeze(0)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                mixture = resampler(mixture)
                source1 = resampler(source1)
                source2 = resampler(source2)
            
            # Trim to max_samples to prevent OOM
            if mixture.shape[0] > self.max_samples:
                mixture = mixture[:self.max_samples]
                source1 = source1[:self.max_samples]
                source2 = source2[:self.max_samples]
            
            # Get speaker IDs from metadata
            speaker1_id = self.metadata[idx, 1]
            speaker2_id = self.metadata[idx, 2]
            
            return {
                'mixture_id': mixture_id,
                'mixture': mixture,
                'source1': source1,
                'source2': source2,
                'speaker1_id': speaker1_id,
                'speaker2_id': speaker2_id
            }
        except Exception as e:
            print(f"Error loading audio {mixture_id}: {e}")
            # Return a dummy sample of acceptable length
            dummy_audio = torch.zeros(self.max_samples)
            return {
                'mixture_id': f"error_{mixture_id}",
                'mixture': dummy_audio,
                'source1': dummy_audio.clone(),
                'source2': dummy_audio.clone(),
                'speaker1_id': "unknown",
                'speaker2_id': "unknown"
            }

def custom_collate(batch):
    """
    Custom collate function to handle variable length audio.
    It finds the minimum length across all audio in the batch and trims.
    """
    try:
        # Extract individual elements from batch
        mixture_ids = [item['mixture_id'] for item in batch]
        mixtures = [item['mixture'] for item in batch]
        source1s = [item['source1'] for item in batch]
        source2s = [item['source2'] for item in batch]
        speaker1_ids = [item['speaker1_id'] for item in batch]
        speaker2_ids = [item['speaker2_id'] for item in batch]
        
        # Get minimum length across all audio files
        min_length = min([
            min([m.size(-1) for m in mixtures]),
            min([s1.size(-1) for s1 in source1s]),
            min([s2.size(-1) for s2 in source2s])
        ])
        
        # Trim all audio to the same length
        mixtures = [m[..., :min_length] for m in mixtures]
        source1s = [s1[..., :min_length] for s1 in source1s]
        source2s = [s2[..., :min_length] for s2 in source2s]
        
        # Stack trimmed audio
        mixtures = torch.stack(mixtures, dim=0)
        source1s = torch.stack(source1s, dim=0)
        source2s = torch.stack(source2s, dim=0)
        
        return {
            'mixture_id': mixture_ids,
            'mixture': mixtures,
            'source1': source1s,
            'source2': source2s,
            'speaker1_id': speaker1_ids,
            'speaker2_id': speaker2_ids
        }
    except Exception as e:
        print(f"ERROR in custom_collate: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a minimal batch with zero tensors in case of error
        batch_size = len(batch)
        return {
            'mixture_id': [f"error_{i}" for i in range(batch_size)],
            'mixture': torch.zeros((batch_size, 16000)),
            'source1': torch.zeros((batch_size, 16000)),
            'source2': torch.zeros((batch_size, 16000)),
            'speaker1_id': ["unknown"] * batch_size,
            'speaker2_id': ["unknown"] * batch_size
        }

def train_generator(generator, discriminator, optimizer_g, mixtures, sources, si_sdr_loss, perceptual_loss, feature_extractor, lambda_perceptual=0.1, lambda_id=0.1, return_loss_tensors=False):
    """Train the generator (SepFormer) to minimize separation loss + perceptual loss + id loss."""
    # Only zero gradients if optimizer is provided (otherwise handled by caller)
    if optimizer_g is not None:
        optimizer_g.zero_grad()
    
    # Forward through generator (on GPU 0)
    estimated_sources = generator(mixtures)
    
    # Ensure the tensor dimensions match our expected format
    # SepFormer returns [B, T, C], we need [B, C, T]
    if estimated_sources.shape[-1] == 2:  # If the last dimension is the number of sources
        estimated_sources = estimated_sources.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
    
    # Calculate separation loss (on GPU 0)
    separation_loss = si_sdr_loss(estimated_sources, sources)
    
    # Initialize loss components
    total_loss = separation_loss
    perceptual_loss_value = torch.tensor(0.0, device=GENERATOR_DEVICE)
    id_loss_value = torch.tensor(0.0, device=GENERATOR_DEVICE)
    
    # Use perceptual loss if lambda > 0
    if lambda_perceptual > 0:
        # Clone tensors to avoid modifying originals
        sources_clone = sources.clone()
        estimated_sources_clone = estimated_sources.clone()
        
        # Apply perceptual loss (already on correct device)
        perceptual_loss_value = perceptual_loss(estimated_sources_clone, sources_clone)
        total_loss = total_loss + lambda_perceptual * perceptual_loss_value
    
    # Use id loss if lambda > 0
    if lambda_id > 0:
        batch_size, num_sources, _ = estimated_sources.shape
        
        # Process each separated source for speaker id
        for i in range(num_sources):
            # Extract source from batch
            est_source_i = estimated_sources[:, i, :]  # [B, T]
            
            # Clone the tensor before processing
            est_source_i_cpu = est_source_i.clone()
            
            # Move to CPU for feature extraction and detach from computation graph
            est_source_i_cpu = est_source_i_cpu.cpu().detach()
            
            # Process with feature extractor
            inputs = feature_extractor(
                est_source_i_cpu.numpy(), 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            ).input_values
            
            # Move to discriminator device
            est_source_i_processed = inputs.to(DISCRIMINATOR_DEVICE)
            
            # Forward through discriminator (on GPU 1)
            pred = discriminator(est_source_i_processed)
            
            # Target index corresponds to source index
            target_idx = i
            targets = target_idx * torch.ones(batch_size, dtype=torch.long).to(DISCRIMINATOR_DEVICE)
            
            # Calculate CE loss
            ce_loss = F.cross_entropy(pred, targets)
            
            # Move loss to generator device
            id_loss_value += ce_loss.to(GENERATOR_DEVICE)
        
        # Average over sources
        id_loss_value = id_loss_value / num_sources
        
        # Add to total loss
        total_loss = total_loss + lambda_id * id_loss_value
    
    # Backward and optimize only if optimizer is provided
    if optimizer_g is not None:
        total_loss.backward()
        optimizer_g.step()
    
    # For AMP, we need to return the actual tensor for backward
    if return_loss_tensors:
        return {
            'total_loss': total_loss,
            'separation_loss': separation_loss,
            'perceptual_loss': perceptual_loss_value,
            'id_loss': id_loss_value
        }
    
    # Return metrics (as scalars to avoid keeping the computation graph)
    return {
        'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'separation_loss': separation_loss.item() if isinstance(separation_loss, torch.Tensor) else separation_loss,
        'perceptual_loss': perceptual_loss_value.item() if isinstance(perceptual_loss_value, torch.Tensor) else perceptual_loss_value,
        'id_loss': id_loss_value.item() if isinstance(id_loss_value, torch.Tensor) else id_loss_value
    }

def process_audio_for_wavlm(audio, feature_extractor, device):
    """Process audio through the feature extractor for WavLM input."""
    try:
        # Clone and detach the tensor to avoid modifying the original and remove gradients
        audio_cpu = audio.cpu().detach().clone()
        
        # Process with feature extractor
        inputs = feature_extractor(
            audio_cpu.numpy(), 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        ).input_values.to(device)
        
        return inputs
    except Exception as e:
        print(f"ERROR in process_audio_for_wavlm: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a dummy tensor in case of error
        return torch.zeros((audio.shape[0], 16000), device=device)

def save_model(model, path):
    """Save model to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_lora_checkpoint(generator, discriminator, path_prefix, optimizer_g=None, optimizer_d=None):
    """Save LoRA weights for both models."""
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    
    # Save SepFormer LoRA weights
    generator_path = f"{path_prefix}_generator_lora.pt"
    save_custom_lora_weights(generator, generator_path)
    print(f"Generator LoRA weights saved to {generator_path}")
    
    # For discriminator using PEFT library
    discriminator_path = f"{path_prefix}_discriminator_lora"
    if hasattr(discriminator, 'save_pretrained'):
        discriminator.save_pretrained(discriminator_path)
        print(f"Discriminator LoRA weights saved to {discriminator_path}")
    else:
        # Fallback to saving full model if needed
        discriminator_path = f"{path_prefix}_discriminator_full.pt"
        # Move the state dict to CPU before saving to avoid GPU memory issues
        discriminator_state = {k: v.detach().cpu() for k, v in discriminator.state_dict().items()}
        torch.save(discriminator_state, discriminator_path)
        print(f"Full discriminator model saved to {discriminator_path}")
    
    # Save combined checkpoint with all state dicts (moved to CPU) if optimizers are provided
    if optimizer_g is not None and optimizer_d is not None:
        try:
            epoch_num = int(path_prefix.split('_')[-1]) if '_epoch_' in path_prefix else args.num_epochs
        except (ValueError, IndexError):
            epoch_num = 0 if 'final' not in path_prefix else args.num_epochs
            
        combined_checkpoint = {
            'epoch': epoch_num,
            'generator_state_dict': {k: v.detach().cpu() for k, v in generator.state_dict().items()},
            'discriminator_state_dict': {k: v.detach().cpu() for k, v in discriminator.state_dict().items()},
            'optimizer_g_state_dict': {k: v if not isinstance(v, torch.Tensor) else v.cpu() 
                                     for k, v in optimizer_g.state_dict().items()},
            'optimizer_d_state_dict': {k: v if not isinstance(v, torch.Tensor) else v.cpu() 
                                     for k, v in optimizer_d.state_dict().items()},
        }
        combined_path = f"{path_prefix}_combined.pt"
        torch.save(combined_checkpoint, combined_path)
        print(f"Combined checkpoint saved to {combined_path}")
    
    return generator_path, discriminator_path

def load_model(model, path):
    """Load model from disk."""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def save_losses(losses, path):
    """Plot and save loss curves."""
    plt.figure(figsize=(12, 8))
    
    # Plot each loss type
    for loss_name, values in losses.items():
        if len(values) > 0:  # Only plot if we have values
            plt.plot(values, label=loss_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training Losses')
    plt.savefig(path)
    plt.close()
    print(f"Loss plot saved to {path}")

def evaluate_model(generator, discriminator, wavlm_model, feature_extractor, test_loader, device_g, device_d):
    """Evaluate the model on the test set."""
    generator.eval()
    discriminator.eval()
    
    metrics = {
        'si_sdr': [],
        'speaker_accuracy': []
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to generator device
            mixtures = batch['mixture'].to(device_g)  # [B, T]
            source1 = batch['source1'].to(device_g)  # [B, T]
            source2 = batch['source2'].to(device_g)  # [B, T]
            
            # Stack sources to create [B, 2, T] tensor
            sources = torch.stack([source1, source2], dim=1)  # [B, 2, T]
            
            # Forward through generator
            estimated_sources = generator(mixtures)
            
            # Ensure the tensor dimensions match our expected format
            # SepFormer returns [B, T, C], we need [B, C, T]
            if estimated_sources.shape[-1] == 2:  # If the last dimension is the number of sources
                estimated_sources = estimated_sources.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
            
            # Calculate SI-SDR
            si_sdr_loss = SI_SDRLoss()
            si_sdr = -si_sdr_loss(estimated_sources, sources).item()  # Negative because it's a loss
            metrics['si_sdr'].append(si_sdr)
            
            # Calculate speaker identification accuracy
            batch_size, num_sources, _ = estimated_sources.shape
            correct = 0
            total = 0
            
            for i in range(num_sources):
                # Extract single source from batch
                est_source_i = estimated_sources[:, i, :]  # [batch_size, samples]
                
                # Process for discriminator (CPU -> GPU 1)
                est_source_i_cpu = est_source_i.cpu().detach().clone()
                inputs = feature_extractor(
                    est_source_i_cpu.numpy(), 
                    sampling_rate=SAMPLE_RATE, 
                    return_tensors="pt", 
                    padding=True
                ).input_values.to(device_d)
                
                # Get speaker prediction from discriminator (on GPU 1)
                pred = discriminator(inputs)
                
                # Target index corresponds to source index 
                target_idx = i
                targets = target_idx * torch.ones(batch_size, dtype=torch.long).to(device_d)
                
                # Count correct predictions
                predicted = torch.argmax(pred, dim=1)
                correct += (predicted == targets).sum().item()
                total += batch_size
            
            metrics['speaker_accuracy'].append(correct / total)
    
    # Average metrics
    for key in metrics:
        metrics[key] = sum(metrics[key]) / len(metrics[key])
    
    return metrics

def main(args):
    # Set random seed
    setup_seed(args.seed)
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set memory allocation config for better fragmentation handling
    try:
        # Try using the newer PyTorch API if available
        if hasattr(torch.cuda, 'set_allocator_settings'):
            print("Setting CUDA memory allocation config to handle fragmentation via API")
            torch.cuda.set_allocator_settings('expandable_segments:True')
        else:
            # Fall back to setting environment variable for older PyTorch versions
            print("Setting CUDA memory allocation config via environment variable")
            # Note: os is already imported at the top of the file
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # This will affect any NEW allocators created after this point
            torch.cuda.empty_cache()  # Force some reinitialization
    except Exception as e:
        print(f"Warning: Could not set memory allocator settings: {e}")
        print("Continuing without memory optimization - you may encounter OOM errors")
    
    # Load models
    print(f"Loading WavLM model: {args.wavlm_model}")
    # Place WavLM on GPU 1
    wavlm_model, feature_extractor = load_wavlm_model(args.wavlm_model, DISCRIMINATOR_DEVICE)
    
    # Create speaker encoder (on GPU 1)
    speaker_encoder = SpeakerEncoder(wavlm_model, embedding_dim=256).to(DISCRIMINATOR_DEVICE)
    
    # Create discriminator (on GPU 1)
    discriminator = SpeakerDiscriminator(speaker_encoder).to(DISCRIMINATOR_DEVICE)
    
    # Apply LoRA to the discriminator (WavLM-based model)
    print("Applying LoRA to the discriminator (WavLM)...")
    discriminator = create_lora_model(discriminator, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    
    # Initialize SepFormer (generator) on GPU 0
    print("Initializing SepFormer model...")
    generator = initialize_sepformer(freeze_params=False).to(GENERATOR_DEVICE)
    
    # Print SepFormer model structure
    print("SepFormer model structure:")
    print(generator)
    
    print("Generator parameters:")
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters before LoRA: {trainable_params:,}")
    
    # Always apply LoRA to SepFormer to prevent OOM issues
    print("Applying LoRA to SepFormer to prevent OOM issues")
    generator = setup_sepformer_lora_for_pipeline(
        generator,
        lora_rank=args.sepformer_lora_rank,  # Use the smaller rank from command line
        lora_alpha=SEPFORMER_LORA_ALPHA
    )
    
    # Print updated parameter counts
    lora_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Trainable parameters after LoRA: {lora_params:,}")
    print(f"Parameter reduction: {trainable_params / max(1, lora_params):.2f}x")
    
    # Make sure discriminator LoRA parameters are trainable
    for name, param in discriminator.named_parameters():
        if 'lora' in name or 'classifier' in name:
            param.requires_grad = True
    
    # Create optimizers - only train parameters that require gradients
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_g, weight_decay=0.0)
    optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_d, weight_decay=0.0)
    
    # Create loss functions
    si_sdr_loss = SI_SDRLoss().to(GENERATOR_DEVICE)
    
    # Use the simpler PerceptualLoss instead of ModelBasedPerceptualLoss to avoid device issues
    perceptual_loss = PerceptualLoss(
        feature_extractor=feature_extractor, 
        sample_rate=16000
    ).to(GENERATOR_DEVICE)
    
    print(f"Using simpler PerceptualLoss to avoid cross-device gradient issues")
    
    # Create GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if args.use_amp:
        print("Using mixed precision training with amp")
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')  # Load to CPU first
        
        # Load generator weights to GPU 0
        generator_state_dict = {k: v.to(GENERATOR_DEVICE) for k, v in checkpoint['generator_state_dict'].items()}
        generator.load_state_dict(generator_state_dict)
        
        # Load discriminator weights to GPU 1
        discriminator_state_dict = {k: v.to(DISCRIMINATOR_DEVICE) for k, v in checkpoint['discriminator_state_dict'].items()}
        discriminator.load_state_dict(discriminator_state_dict)
        
        if not args.evaluate:  # Only load optimizer states if training
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            if 'scaler' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
    
    # Create datasets with limited duration to prevent OOM
    print(f"Loading datasets from {args.data_dir}")
    train_dataset = Vox2MixDataset(args.data_dir, split="train", sample_rate=SAMPLE_RATE, max_duration=args.max_duration)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate)
    
    test_dataset = Vox2MixDataset(args.data_dir, split="test", sample_rate=SAMPLE_RATE, max_duration=args.max_duration)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)
    
    # Test mode - run a single batch through and exit
    if args.test_mode:
        print("RUNNING IN TEST MODE - Processing a single batch")
        for batch in train_loader:
            print("=" * 50)
            print("Got batch from loader, processing...")
            
            # Move data to generator device
            mixtures = batch['mixture'].to(GENERATOR_DEVICE)  # [B, T]
            source1 = batch['source1'].to(GENERATOR_DEVICE)  # [B, T]
            source2 = batch['source2'].to(GENERATOR_DEVICE)  # [B, T]
            
            # Stack sources to create [B, 2, T] tensor
            sources = torch.stack([source1, source2], dim=1)  # [B, 2, T]
            
            print(f"Mixtures shape: {mixtures.shape}")
            print(f"Sources shape: {sources.shape}")
            
            # Get details about the SepFormer configuration
            try:
                print("\nSepFormer configuration:")
                if hasattr(generator, 'hparams'):
                    print(f"Number of speakers: {generator.hparams.num_spks}")
            except Exception as e:
                print(f"Error getting SepFormer configuration: {e}")
            
            # Test SepFormer
            print("\nTesting SepFormer forward pass...")
            with torch.no_grad():
                try:
                    # Forward through generator
                    estimated_sources = generator(mixtures)
                    print(f"SepFormer output shape: {estimated_sources.shape}")
                    
                    # Permute if needed
                    if estimated_sources.shape[-1] == 2:  # If the last dimension is the number of sources
                        estimated_sources = estimated_sources.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
                        print(f"SepFormer permuted output shape: {estimated_sources.shape}")
                    
                    # Test SI-SDR loss
                    print("\nTesting SI-SDR loss...")
                    sdr_loss = si_sdr_loss(estimated_sources, sources)
                    print(f"SI-SDR loss: {sdr_loss.item()}")
                    
                    # Test perceptual loss
                    print("\nTesting perceptual loss...")
                    p_loss = perceptual_loss(estimated_sources, sources)
                    print(f"Perceptual loss: {p_loss.item()}")
                    
                    # Test discriminator
                    print("\nTesting discriminator...")
                    for i in range(sources.shape[1]):
                        # Process source through discriminator
                        src = estimated_sources[:, i, :]
                        # Move to CPU then to discriminator device
                        src_cpu = src.cpu().detach()
                        src_processed = feature_extractor(
                            src_cpu.numpy(), 
                            sampling_rate=SAMPLE_RATE, 
                            return_tensors="pt", 
                            padding=True
                        ).input_values.to(DISCRIMINATOR_DEVICE)
                        
                        pred = discriminator(src_processed)
                        print(f"Discriminator output for source {i}: shape={pred.shape}")
                        
                    print("\nAll components working correctly!")
                except Exception as e:
                    print(f"ERROR during test mode: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print("\nTest completed, exiting...")
            break
        return
    elif args.evaluate:
        # Evaluation mode
        print("Running evaluation...")
        metrics = evaluate_model(generator, discriminator, wavlm_model, feature_extractor, test_loader, GENERATOR_DEVICE, DISCRIMINATOR_DEVICE)
        print("Evaluation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        # Training mode
        # Training loop
        for epoch in range(start_epoch, args.num_epochs):
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            generator.train()
            discriminator.train()
            
            # Progress bar
            pbar = tqdm(train_loader)
            
            # Metrics for this epoch
            epoch_metrics = {
                'total_loss': 0.0,
                'separation_loss': 0.0,
                'perceptual_loss': 0.0,
                'id_loss': 0.0
            }
            
            # For short training mode, only process a few steps
            max_steps = 10 if args.short_train else len(train_loader)
            
            # Set up gradient accumulation
            optimizer_g.zero_grad()
            accumulated_samples = 0
            
            for i, batch in enumerate(pbar):
                # Move data to generator device
                mixtures = batch['mixture'].to(GENERATOR_DEVICE)  # [B, T]
                source1 = batch['source1'].to(GENERATOR_DEVICE)  # [B, T]
                source2 = batch['source2'].to(GENERATOR_DEVICE)  # [B, T]
                
                # Stack sources to create [B, 2, T] tensor
                sources = torch.stack([source1, source2], dim=1)  # [B, 2, T]
                
                # Get current batch size and update accumulated samples
                curr_batch_size = mixtures.size(0)
                accumulated_samples += curr_batch_size
                
                # Skip tiny batches (less than 1s of audio) - they're often errors
                if mixtures.size(1) < SAMPLE_RATE:
                    print(f"Skipping tiny batch: {mixtures.size()}")
                    continue
                
                # Train generator
                try:
                    # Use mixed precision for forward/backward if enabled
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            g_metrics = train_generator(
                                generator=generator, 
                                discriminator=discriminator, 
                                optimizer_g=None,  # Don't step optimizer now - we'll do it after accumulation
                                mixtures=mixtures, 
                                sources=sources, 
                                si_sdr_loss=si_sdr_loss, 
                                perceptual_loss=perceptual_loss, 
                                feature_extractor=feature_extractor,
                                lambda_perceptual=0.1, 
                                lambda_id=0.1,
                                return_loss_tensors=True
                            )
                            
                            # Scale the loss by batch_size / effective_batch_size for proper gradients
                            scaled_loss = g_metrics['total_loss'] * (curr_batch_size / args.effective_batch_size)
                            
                        # Scale gradients and accumulate
                        scaler.scale(scaled_loss).backward()
                        
                        # Step optimizer if we've accumulated enough samples
                        if accumulated_samples >= args.effective_batch_size or i == len(train_loader) - 1:
                            scaler.step(optimizer_g)
                            scaler.update()
                            optimizer_g.zero_grad()
                            accumulated_samples = 0
                    else:
                        # Standard precision training
                        g_metrics = train_generator(
                            generator=generator, 
                            discriminator=discriminator, 
                            optimizer_g=None,  # Don't step optimizer now
                            mixtures=mixtures, 
                            sources=sources, 
                            si_sdr_loss=si_sdr_loss, 
                            perceptual_loss=perceptual_loss, 
                            feature_extractor=feature_extractor,
                            lambda_perceptual=0.1, 
                            lambda_id=0.1,
                            return_loss_tensors=True
                        )
                        
                        # Scale the loss for accumulation
                        scaled_loss = g_metrics['total_loss'] * (curr_batch_size / args.effective_batch_size)
                        scaled_loss.backward()
                        
                        # Step optimizer if we've accumulated enough samples
                        if accumulated_samples >= args.effective_batch_size or i == len(train_loader) - 1:
                            optimizer_g.step()
                            optimizer_g.zero_grad()
                            accumulated_samples = 0
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"OOM error processing batch {i} with shape {mixtures.shape}, skipping...")
                        # Skip this batch and clear memory
                        if args.use_amp:
                            scaler.update()
                        optimizer_g.zero_grad()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                # Update and display metrics
                for k, v in g_metrics.items():
                    # If v is a tensor, convert to scalar for logging
                    if isinstance(v, torch.Tensor):
                        epoch_metrics[k] += v.item()
                    else:
                        epoch_metrics[k] += v
                
                # Update progress bar with detailed loss components
                # Extract scalar values for display
                total_loss_val = g_metrics['total_loss'].item() if isinstance(g_metrics['total_loss'], torch.Tensor) else g_metrics['total_loss']
                sep_loss_val = g_metrics['separation_loss'].item() if isinstance(g_metrics['separation_loss'], torch.Tensor) else g_metrics['separation_loss']
                perc_loss_val = g_metrics['perceptual_loss'].item() if isinstance(g_metrics['perceptual_loss'], torch.Tensor) else g_metrics['perceptual_loss']
                id_loss_val = g_metrics['id_loss'].item() if isinstance(g_metrics['id_loss'], torch.Tensor) else g_metrics['id_loss']
                
                pbar.set_description(
                    f"Total: {total_loss_val:.4f} | Sep: {sep_loss_val:.4f} | Perc: {perc_loss_val:.4f} | ID: {id_loss_val:.4f}"
                )
                
                # Periodic memory cleanup
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Break if short training mode
                if args.short_train and i >= max_steps - 1:
                    print(f"Short training mode: stopping after {i+1} steps")
                    break
            
            # Free cache after each epoch
            torch.cuda.empty_cache()
            
            # Average metrics for this epoch
            num_steps = min(max_steps, len(train_loader))
            for k in epoch_metrics:
                epoch_metrics[k] /= num_steps
            
            print(f"Epoch {epoch+1} metrics:")
            for k, v in epoch_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Run evaluation on test set (skip for short training)
            if not args.short_train:
                print("Evaluating on test set...")
                eval_metrics = evaluate_model(generator, discriminator, wavlm_model, feature_extractor, test_loader, GENERATOR_DEVICE, DISCRIMINATOR_DEVICE)
                print("Test metrics:")
                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}")
            else:
                # For short training, use dummy metrics
                eval_metrics = {'si_sdr': 0.0, 'speaker_accuracy': 0.0}
                print("Skipping evaluation in short training mode")
            
            # Save model checkpoint
            checkpoint_path = os.path.join(args.results_dir, f"checkpoint_epoch_{epoch+1}")
            
            # Save all checkpoint data
            save_lora_checkpoint(generator, discriminator, checkpoint_path, optimizer_g, optimizer_d)
            
            # If using mixed precision, also save the scaler state
            if args.use_amp and scaler is not None:
                # Save additional info in the checkpoint
                combined_checkpoint = torch.load(f"{checkpoint_path}_combined.pt")
                combined_checkpoint['scaler'] = scaler.state_dict()
                combined_checkpoint['train_metrics'] = epoch_metrics
                combined_checkpoint['eval_metrics'] = eval_metrics
                torch.save(combined_checkpoint, f"{checkpoint_path}_combined.pt")
            
            print(f"Saved checkpoint for epoch {epoch+1}")
            
            # Break after one epoch in short training mode
            if args.short_train:
                print("Short training mode: stopping after one epoch")
                break
        
        # Save final models
        final_checkpoint_path = os.path.join(args.results_dir, "final_checkpoint")
        save_lora_checkpoint(generator, discriminator, final_checkpoint_path, optimizer_g, optimizer_d)
        print("Training completed and final models saved")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune combined model with LoRA")
    parser.add_argument("--data_dir", type=str, default=VOX2MIX_DIR, help="Directory containing mixed utterances")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--wavlm_model", type=str, default=WAVLM_MODEL_NAME, help="WavLM model name")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr_g", type=float, default=LEARNING_RATE, help="Learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=LEARNING_RATE_D, help="Learning rate for discriminator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model instead of training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode (process a single batch and exit)")
    parser.add_argument("--short_train", action="store_true", help="Run a short training session (10 steps max)")
    parser.add_argument("--sepformer_lora_rank", type=int, default=2, help="LoRA rank for SepFormer")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--effective_batch_size", type=int, default=16000, help="Effective batch size for gradient accumulation")
    parser.add_argument("--max_duration", type=float, default=8.0, help="Maximum duration of audio samples")
    args = parser.parse_args()
    
    main(args)
