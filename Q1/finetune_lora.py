#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchaudio
import random
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.data_utils import VoxCelebDataset, get_vox2_train_test_datasets
from src.model_utils import load_wavlm_model, create_lora_model, ArcFaceLayer
from src.evaluation import cosine_similarity, compute_eer

# Paths and constants
VOX2_DIR = "/home/m23mac008/assignment_2_m23mac008/q1/Speech2025Datasets/vox2"
MODEL_NAME = "microsoft/wavlm-base-plus"
BATCH_SIZE = 4  # Smaller batch size for CPU
NUM_EPOCHS = 10  # Epochs for training
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 256
NUM_TRAIN_SPEAKERS = 100  # Use all available speakers
SAVE_DIR = "models"

class SpeakerVerificationModel(nn.Module):
    def __init__(self, wavlm_model, num_speakers, embedding_dim=256):
        super(SpeakerVerificationModel, self).__init__()
        self.wavlm = wavlm_model
        self.embedding_dim = embedding_dim
        
        # Projection layer for speaker embeddings
        self.projection = nn.Linear(768, embedding_dim)  # WavLM base hidden size is 768
        
        # ArcFace classifier
        self.arc_face = ArcFaceLayer(embedding_dim, num_speakers)
    
    def forward(self, x, labels=None):
        # Get WavLM representations
        outputs = self.wavlm(x)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling over time dimension
        pooled = torch.mean(hidden_states, dim=1)
        
        # Project to embedding space
        embeddings = self.projection(pooled)
        
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # If training, return logits through ArcFace
        if labels is not None:
            logits = self.arc_face(embeddings, labels)
            return logits, embeddings
        
        # If just extracting features, return embeddings
        return embeddings

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load pre-trained WavLM model
    base_model, feature_extractor = load_wavlm_model(MODEL_NAME, device)
    
    # Print model module names for debugging
    print("Available modules in WavLM model:")
    module_names = []
    for name, _ in base_model.named_modules():
        if len(name.split('.')) > 1:  # Skip top-level modules
            module_names.append(name)
    
    # Print a sample of module names
    for name in sorted(module_names)[:20]:
        print(f"  - {name}")
    
    # Apply LoRA to the base model
    lora_model = create_lora_model(base_model)
    
    # Get all speakers without splitting into train/test
    train_speakers, _ = get_vox2_train_test_datasets(VOX2_DIR, NUM_TRAIN_SPEAKERS)
    
    # Create dataset for the entire VOX2 directory
    print("Creating dataset...")
    dataset = VoxCelebDataset(VOX2_DIR)
    
    # Filter to include all available speakers
    train_samples = []
    for sample in dataset.samples:
        if sample['speaker_dir'] in train_speakers:
            train_samples.append(sample)
    
    # For testing purposes, limit the number of samples
    MAX_SAMPLES_PER_SPEAKER = 50  # Limit samples per speaker for faster testing
    
    # Filter samples to limit per speaker
    filtered_samples = []
    speaker_counts = {}
    
    for sample in train_samples:
        speaker = sample['speaker_dir']
        if speaker not in speaker_counts:
            speaker_counts[speaker] = 0
        
        if speaker_counts[speaker] < MAX_SAMPLES_PER_SPEAKER:
            filtered_samples.append(sample)
            speaker_counts[speaker] += 1
    
    # Use the filtered samples
    train_samples = filtered_samples
    
    print(f"Limited to {MAX_SAMPLES_PER_SPEAKER} samples per speaker for testing")
    
    # Remap speaker IDs to consecutive integers for ArcFace
    speaker_id_map = {}
    for i, speaker in enumerate(train_speakers):
        speaker_id_map[speaker] = i
    
    # Update speaker IDs in the training samples
    for sample in train_samples:
        sample['speaker_id'] = speaker_id_map[sample['speaker_dir']]
    
    # Create custom datasets from the filtered samples
    class FilteredDataset(Dataset):
        def __init__(self, dataset, samples):
            self.dataset = dataset
            self.samples = samples
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            # Get the original sample information 
            original_sample = self.samples[idx]
            
            # Load the audio data directly
            waveform, sample_rate = torchaudio.load(original_sample['audio_path'])
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed (assuming 16000 Hz sample rate as in VoxCelebDataset)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Ensure consistent length (assuming 5.0 seconds max duration as in VoxCelebDataset)
            max_length = int(5.0 * 16000)
            if waveform.shape[1] > max_length:
                # Take a random crop
                start = random.randint(0, waveform.shape[1] - max_length)
                waveform = waveform[:, start:start+max_length]
            else:
                # Pad with zeros
                padding = max_length - waveform.shape[1]
                if padding > 0:
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return {
                'waveform': waveform,
                'speaker_id': original_sample['speaker_id'],
                'speaker_dir': original_sample['speaker_dir']
            }
    
    train_filtered_dataset = FilteredDataset(dataset, train_samples)
    
    train_loader = DataLoader(train_filtered_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Train dataset size: {len(train_filtered_dataset)}")
    
    # Create the full model with ArcFace
    num_speakers = len(speaker_id_map)  # Use the actual number of training speakers
    print(f"Number of speakers for ArcFace: {num_speakers}")
    model = SpeakerVerificationModel(lora_model, num_speakers, EMBEDDING_DIM).to(device)
    
    # Define optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    verification_aucs = []
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        # Use tqdm with additional metrics display
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(train_progress_bar):
            waveforms = batch['waveform'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            
            # Process inputs with feature extractor
            inputs = feature_extractor(
                waveforms.squeeze(1).cpu().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(device)
            
            # Forward pass
            logits, _ = model(inputs, speaker_ids)
            loss = criterion(logits, speaker_ids)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar with current loss
            train_progress_bar.set_postfix({
                'loss': f"{total_train_loss / train_batches:.4f}"
            })
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Verification check on a subset of training data
        model.eval()
        
        # Perform verification check on training data
        if len(train_samples) > 100:
            verification_subset = random.sample(train_samples, 100)
            
            # Create speaker pairs for verification
            verification_pairs = []
            speaker_samples = {}
            
            # Group samples by speaker
            for sample in verification_subset:
                spk = sample['speaker_dir']
                if spk not in speaker_samples:
                    speaker_samples[spk] = []
                speaker_samples[spk].append(sample)
            
            # Create same-speaker pairs
            for spk, samples in speaker_samples.items():
                if len(samples) >= 2:
                    pairs = min(2, len(samples) // 2)  # Up to 2 pairs per speaker
                    for _ in range(pairs):
                        s1, s2 = random.sample(samples, 2)
                        verification_pairs.append((s1, s2, 1))  # 1 = same speaker
            
            # Create different-speaker pairs
            speakers = list(speaker_samples.keys())
            if len(speakers) >= 2:
                for _ in range(min(20, len(verification_pairs) * 2)):
                    spk1, spk2 = random.sample(speakers, 2)
                    s1 = random.choice(speaker_samples[spk1])
                    s2 = random.choice(speaker_samples[spk2])
                    verification_pairs.append((s1, s2, 0))  # 0 = different speaker
            
            if verification_pairs:
                # Compute verification scores
                scores = []
                labels = []
                
                with torch.no_grad():
                    for s1, s2, label in verification_pairs:
                        # Process first sample
                        waveform1, _ = torchaudio.load(s1['audio_path'])
                        if waveform1.shape[0] > 1:
                            waveform1 = torch.mean(waveform1, dim=0, keepdim=True)
                        
                        # Process second sample
                        waveform2, _ = torchaudio.load(s2['audio_path'])
                        if waveform2.shape[0] > 1:
                            waveform2 = torch.mean(waveform2, dim=0, keepdim=True)
                        
                        # Extract embeddings
                        inputs1 = feature_extractor(
                            waveform1.cpu().numpy(), 
                            sampling_rate=16000, 
                            return_tensors="pt", 
                            padding=True
                        ).input_values.to(device)
                        
                        inputs2 = feature_extractor(
                            waveform2.cpu().numpy(), 
                            sampling_rate=16000, 
                            return_tensors="pt", 
                            padding=True
                        ).input_values.to(device)
                        
                        emb1 = model(inputs1)
                        emb2 = model(inputs2)
                        
                        # Compute cosine similarity
                        similarity = F.cosine_similarity(emb1, emb2)
                        scores.append(similarity.item())
                        labels.append(label)
                
                if len(scores) > 0 and len(set(labels)) > 1:
                    # Compute verification AUC
                    auc = roc_auc_score(labels, scores)
                    verification_aucs.append(auc)
                    print(f"\nVerification check - AUC: {auc:.4f}")
                    
                    # Print score statistics
                    same_scores = [s for s, l in zip(scores, labels) if l == 1]
                    diff_scores = [s for s, l in zip(scores, labels) if l == 0]
                    
                    if same_scores:
                        print(f"  Same speaker scores - avg: {np.mean(same_scores):.4f}, min: {min(same_scores):.4f}, max: {max(same_scores):.4f}")
                    if diff_scores:
                        print(f"  Different speaker scores - avg: {np.mean(diff_scores):.4f}, min: {min(diff_scores):.4f}, max: {max(diff_scores):.4f}")
                    
                    # Use verification AUC as a model selection metric
                    current_metric = 1.0 - auc  # Lower is better (like a loss)
                else:
                    current_metric = avg_train_loss  # Fall back to train loss
            else:
                current_metric = avg_train_loss  # Fall back to train loss
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        if len(verification_aucs) > 0:
            print(f"  Verification AUC: {verification_aucs[-1]:.4f}")
        
        # Save model if metric improved
        if current_metric < best_loss:
            best_loss = current_metric
            print(f"  New best model! Saving checkpoint...")
            # Save only the state dict
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"  Saving regular checkpoint...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            
        print("-" * 50)  # Separator between epochs
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pth'))
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS+1), train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Training Loss')
    
    if len(verification_aucs) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(verification_aucs)+1), verification_aucs)
        plt.xlabel('Epochs')
        plt.ylabel('Verification AUC')
        plt.title('Verification AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curve.png'))
    plt.close()
    
    print("Training complete. Model saved to 'models/' directory.")

if __name__ == "__main__":
    main() 