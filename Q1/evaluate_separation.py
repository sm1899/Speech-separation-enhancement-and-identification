#!/usr/bin/env python3
"""
Speaker Identification After Separation

This script evaluates speaker identification performance on separated audio sources using:
1. Pre-trained WavLM model 
2. Fine-tuned WavLM model with speaker verification architecture

It maps separated sources back to original speakers and reports Rank-1 identification accuracy.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio
import pandas as pd
import time
import argparse

from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.metrics import accuracy_score, confusion_matrix

# Define SpeakerVerificationModel class
class SpeakerVerificationModel(torch.nn.Module):
    def __init__(self, wavlm_model, num_speakers, embedding_dim=256):
        super(SpeakerVerificationModel, self).__init__()
        self.wavlm = wavlm_model
        self.embedding_dim = embedding_dim
        
        # Projection layer for speaker embeddings
        self.projection = torch.nn.Linear(768, embedding_dim)  # WavLM base hidden size is 768
        
        # ArcFace classifier - not used during inference
        self.arc_face = torch.nn.Linear(embedding_dim, num_speakers)
    
    def forward(self, x, labels=None):
        # Get WavLM representations
        outputs = self.wavlm(x)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling over time dimension
        pooled = torch.mean(hidden_states, dim=1)
        
        # Project to embedding space
        embeddings = self.projection(pooled)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # If training, return logits through ArcFace
        if labels is not None:
            logits = self.arc_face(embeddings, labels)
            return logits, embeddings
        
        # If just extracting features, return embeddings
        return embeddings

def load_and_process_audio(audio_path, target_sr=16000):
    """
    Load audio file and ensure it's at the correct sample rate
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        
    return waveform, target_sr

def load_wavlm_model(device=None):
    """
    Load the WavLM base model and feature extractor
    """
    model_name = "microsoft/wavlm-base-plus"
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading WavLM model: {model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name)
    model = model.to(device)
    
    return model, feature_extractor

def compute_embeddings_pretrained(model, feature_extractor, audio_path, device):
    """
    Compute embeddings using pre-trained WavLM model
    """
    # Load and resample audio
    waveform, sr = load_and_process_audio(audio_path)
    
    # Extract features
    inputs = feature_extractor(
        waveform.squeeze(0).cpu().numpy(), 
        sampling_rate=sr, 
        return_tensors="pt", 
        padding=True
    ).input_values.to(device)
    
    # Compute embeddings
    with torch.no_grad():
        outputs = model(inputs)
        hidden_states = outputs.last_hidden_state
        embedding = torch.mean(hidden_states, dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    return embedding

def compute_embeddings_finetuned(model, feature_extractor, audio_path, device):
    """
    Compute embeddings using fine-tuned WavLM model with projection layer
    """
    # Load and resample audio
    waveform, sr = load_and_process_audio(audio_path)
    
    # Extract features
    inputs = feature_extractor(
        waveform.squeeze(0).cpu().numpy(), 
        sampling_rate=sr, 
        return_tensors="pt", 
        padding=True
    ).input_values.to(device)
    
    # Compute embeddings
    with torch.no_grad():
        embedding = model(inputs)
    
    return embedding

def compute_similarity(ref_embedding, sep_embedding):
    """
    Compute cosine similarity between embeddings
    """
    similarity = torch.nn.functional.cosine_similarity(ref_embedding, sep_embedding)
    return similarity.item()

def identify_speaker(separated_embedding, reference_embeddings, reference_speaker_ids):
    """
    Identify the speaker in a separated audio file
    """
    max_similarity = -1.0
    identified_speaker = None
    
    # Compute similarity with each reference embedding
    for idx, ref_embedding in enumerate(reference_embeddings):
        similarity = compute_similarity(ref_embedding, separated_embedding)
        
        if similarity > max_similarity:
            max_similarity = similarity
            identified_speaker = reference_speaker_ids[idx]
    
    return identified_speaker, max_similarity

def evaluate_identification(model, feature_extractor, device, 
                            reference_speakers_dir, separated_dir, metadata_path,
                            is_finetuned=False):
    """
    Evaluate speaker identification performance
    
    Args:
        model: WavLM model (pre-trained or fine-tuned)
        feature_extractor: WavLM feature extractor
        device: Device to run on
        reference_speakers_dir: Directory containing reference speaker recordings
        separated_dir: Directory containing separated sources
        metadata_path: Path to ground truth metadata CSV file
        is_finetuned: Whether the model is fine-tuned
        
    Returns:
        metrics: Dictionary containing identification metrics
    """
    # Load ground truth metadata
    metadata = pd.read_csv(metadata_path)
    
    # Convert mixture_id to string with zero-padding to match file format
    metadata['mixture_id_str'] = metadata['mixture_id'].apply(lambda x: f"{x:04d}")
    
    # Group speakers by ID for reference
    speaker_ids = set()
    for _, row in metadata.iterrows():
        speaker_ids.add(row['speaker1_id'])
        speaker_ids.add(row['speaker2_id'])
    
    speaker_ids = sorted(list(speaker_ids))
    print(f"Found {len(speaker_ids)} unique speakers in metadata")
    
    # Compute reference embeddings for each speaker
    reference_embeddings = []
    reference_speaker_ids = []
    
    for speaker_id in tqdm(speaker_ids, desc="Computing reference embeddings"):
        speaker_dir = os.path.join(reference_speakers_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            print(f"Warning: Speaker directory not found: {speaker_dir}")
            continue
        
        # Find first valid utterance for this speaker
        for session in os.listdir(speaker_dir):
            session_dir = os.path.join(speaker_dir, session)
            if not os.path.isdir(session_dir):
                continue
            
            utterances = [f for f in os.listdir(session_dir) if f.endswith('.wav')]
            if not utterances:
                continue
            
            # Use first utterance as reference
            utterance_path = os.path.join(session_dir, utterances[0])
            
            try:
                if is_finetuned:
                    embedding = compute_embeddings_finetuned(model, feature_extractor, utterance_path, device)
                else:
                    embedding = compute_embeddings_pretrained(model, feature_extractor, utterance_path, device)
                
                reference_embeddings.append(embedding)
                reference_speaker_ids.append(speaker_id)
                break
            except Exception as e:
                print(f"Error processing reference utterance {utterance_path}: {str(e)}")
                continue
    
    print(f"Computed reference embeddings for {len(reference_speaker_ids)} speakers")
    
    # Process separated sources
    separated_files = sorted([f for f in os.listdir(separated_dir) if f.endswith('.wav')])
    
    # Mapping between source ID and separation channel
    source_indices = {"source1": 1, "source2": 2}
    
    # Track identifications and ground truth
    identifications = {}
    correct_identifications = 0
    total_evaluations = 0
    
    for file in tqdm(separated_files, desc="Identifying speakers"):
        try:
            # Extract mixture ID and source number from filename
            # Format is "separated_mixture_XXXX_sourceY.wav"
            parts = file.replace('separated_mixture_', '').replace('.wav', '').split('_')
            
            # The format should be ['XXXX', 'sourceY']
            if len(parts) == 2 and parts[1].startswith('source'):
                # Keep the original ID with padding
                mixture_id = parts[0]  # This preserves zero-padding
                source_str = parts[1]
                source_idx = int(source_str.replace('source', ''))
            else:
                continue
            
            # Skip if source index is not valid
            if source_idx not in [1, 2]:
                continue
            
            # Find this mixture in metadata
            mixture_data = metadata[metadata['mixture_id_str'] == mixture_id]
            
            if mixture_data.empty:
                continue
            
            # Get ground truth speaker ID
            ground_truth_speaker = mixture_data.iloc[0][f'speaker{source_idx}_id']
            
            # Compute embedding for separated source
            source_path = os.path.join(separated_dir, file)
            
            if is_finetuned:
                embedding = compute_embeddings_finetuned(model, feature_extractor, source_path, device)
            else:
                embedding = compute_embeddings_pretrained(model, feature_extractor, source_path, device)
            
            # Identify speaker
            identified_speaker, similarity = identify_speaker(
                embedding, reference_embeddings, reference_speaker_ids
            )
            
            # Store identification result
            source_key = f"{mixture_id}_source{source_idx}"
            identifications[source_key] = {
                'identified_speaker': identified_speaker,
                'ground_truth_speaker': ground_truth_speaker,
                'similarity': similarity,
                'is_correct': identified_speaker == ground_truth_speaker
            }
            
            # Update statistics
            total_evaluations += 1
            if identified_speaker == ground_truth_speaker:
                correct_identifications += 1
            
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    # Calculate metrics
    rank1_accuracy = correct_identifications / total_evaluations if total_evaluations > 0 else 0
    
    metrics = {
        'Rank-1 Accuracy': rank1_accuracy,
        'Correct Identifications': correct_identifications,
        'Total Evaluations': total_evaluations,
        'Number of Reference Speakers': len(reference_speaker_ids)
    }
    
    return metrics, identifications

def plot_confusion_matrix(identifications, reference_speaker_ids, output_path, title="Confusion Matrix"):
    """
    Plot confusion matrix for speaker identification
    """
    # Extract ground truth and predicted labels
    ground_truth = []
    predictions = []
    
    for source_id, data in identifications.items():
        if data['ground_truth_speaker'] in reference_speaker_ids and data['identified_speaker'] in reference_speaker_ids:
            ground_truth.append(data['ground_truth_speaker'])
            predictions.append(data['identified_speaker'])
    
    # Compute confusion matrix
    unique_speakers = sorted(list(set(reference_speaker_ids)))
    
    # If we have predictions, create the confusion matrix
    if ground_truth and predictions:
        cm = confusion_matrix(ground_truth, predictions, labels=unique_speakers)
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=16)
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(unique_speakers))
        short_labels = [s[-4:] for s in unique_speakers]  # Use last 4 chars of ID for readability
        plt.xticks(tick_marks, short_labels, rotation=90, fontsize=8)
        plt.yticks(tick_marks, short_labels, fontsize=8)
        
        # Add values in each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=7)
        
        plt.tight_layout()
        plt.ylabel('True Speaker ID', fontsize=12)
        plt.xlabel('Predicted Speaker ID', fontsize=12)
        
        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    else:
        print("Not enough data to create confusion matrix")

def save_results(identifications, output_path):
    """
    Save identification results to CSV
    """
    df = pd.DataFrame([
        {
            'source_id': source_id,
            'identified_speaker': data['identified_speaker'],
            'ground_truth_speaker': data['ground_truth_speaker'],
            'similarity': data['similarity'],
            'is_correct': data['is_correct']
        }
        for source_id, data in identifications.items()
    ])
    
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    # Paths
    vox2_path = "Speech2025Datasets/vox2"
    separated_dir = "separated_vox2mix/test/separated"
    metadata_path = "vox2mix_dataset/test/metadata.csv"
    output_dir = "separation_identification_results"
    model_path = "models/best_model.pth"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Debug: Check metadata and matchings
    print("\n=== Debug: Checking metadata and files ===")
    
    # Fix for malformed CSV: Read the metadata file manually first
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    
    # Fix line breaks within fields and create a corrected temporary file
    corrected_content = []
    current_line = ""
    for line in lines:
        if line.strip() and ',' in line:
            # If this looks like the start of a new record (has commas)
            if current_line:
                corrected_content.append(current_line)
            current_line = line.strip()
        else:
            # This is a continuation of the previous line
            current_line += line.strip()
    
    # Add the last line
    if current_line:
        corrected_content.append(current_line)
    
    # Create a temporary fixed file
    temp_metadata_path = os.path.join(output_dir, "fixed_metadata.csv")
    with open(temp_metadata_path, 'w') as f:
        f.write("\n".join(corrected_content))
    
    # Now read the fixed CSV file
    metadata = pd.read_csv(temp_metadata_path)
    
    # Convert mixture_id to string with zero-padding to match file format
    metadata['mixture_id_str'] = metadata['mixture_id'].apply(lambda x: f"{x:04d}")
    
    print(f"Metadata contains {len(metadata)} rows")
    
    # Check first few entries
    print("First 5 metadata entries:")
    for i in range(min(5, len(metadata))):
        print(f"  {metadata.iloc[i]['mixture_id']} - Speaker1: {metadata.iloc[i]['speaker1_id']}, Speaker2: {metadata.iloc[i]['speaker2_id']}")
    
    # Check separated files
    separated_files = sorted([f for f in os.listdir(separated_dir) if f.endswith('.wav')])
    print(f"Found {len(separated_files)} separated files")
    
    # Check first few files
    print("First 5 separated files:")
    for i in range(min(5, len(separated_files))):
        print(f"  {separated_files[i]}")
    
    # Try to match a few
    print("Trying to match first 5 separated files to metadata:")
    for i in range(min(5, len(separated_files))):
        file = separated_files[i]
        parts = file.replace('separated_mixture_', '').replace('.wav', '').split('_')
        if len(parts) == 2 and parts[1].startswith('source'):
            # Keep the original ID with padding
            mixture_id = parts[0]  # This preserves zero-padding
            source_str = parts[1]
            source_idx = int(source_str.replace('source', ''))
            
            # Try to match in metadata
            matching_rows = metadata[metadata['mixture_id_str'] == mixture_id]
            match_found = not matching_rows.empty
            
            if match_found:
                matching_row = matching_rows.iloc[0]
                print(f"  File: {file}")
                print(f"    Extracted mixture_id={mixture_id}, source_idx={source_idx}")
                print(f"    Matched to metadata row: {matching_row['mixture_id']} - Speaker{source_idx}: {matching_row[f'speaker{source_idx}_id']}")
            else:
                print(f"  File: {file}")
                print(f"    Extracted mixture_id={mixture_id}, source_idx={source_idx}")
                print(f"    NO MATCH FOUND in metadata")
    
    # Load models
    base_model, feature_extractor = load_wavlm_model(device)
    
    print("\n=== Evaluating Pre-trained WavLM Model ===")
    start_time = time.time()
    
    # Evaluate pre-trained model
    pretrained_metrics, pretrained_identifications = evaluate_identification(
        model=base_model,
        feature_extractor=feature_extractor,
        device=device,
        reference_speakers_dir=vox2_path,
        separated_dir=separated_dir,
        metadata_path=temp_metadata_path,  # Use the fixed metadata path
        is_finetuned=False
    )
    
    pretrained_time = time.time() - start_time
    pretrained_metrics['Elapsed Time'] = pretrained_time
    
    print("\nPre-trained WavLM Results:")
    for key, value in pretrained_metrics.items():
        print(f"  {key}: {value}")
    
    # Save pre-trained results
    save_results(
        pretrained_identifications, 
        os.path.join(output_dir, "pretrained_identifications.csv")
    )
    
    # Create confusion matrix for pre-trained model
    plot_confusion_matrix(
        pretrained_identifications,
        list(set([data['ground_truth_speaker'] for data in pretrained_identifications.values()])),
        os.path.join(output_dir, "pretrained_confusion_matrix.png"),
        "Pre-trained WavLM Speaker Identification"
    )
    
    print("\n=== Evaluating Fine-tuned WavLM Model ===")
    start_time = time.time()
    
    # Create and load fine-tuned model
    NUM_SPEAKERS = 100  # Same as in training
    EMBEDDING_DIM = 256
    print("Creating fine-tuned model instance...")
    finetuned_model = SpeakerVerificationModel(base_model, NUM_SPEAKERS, EMBEDDING_DIM).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            finetuned_model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    else:
        print(f"Warning: Fine-tuned model not found at {model_path}")
    
    print("Starting fine-tuned model evaluation...")
    
    # Evaluate fine-tuned model
    finetuned_metrics, finetuned_identifications = evaluate_identification(
        model=finetuned_model,
        feature_extractor=feature_extractor,
        device=device,
        reference_speakers_dir=vox2_path,
        separated_dir=separated_dir,
        metadata_path=temp_metadata_path,  # Use the fixed metadata path
        is_finetuned=True
    )
    
    finetuned_time = time.time() - start_time
    finetuned_metrics['Elapsed Time'] = finetuned_time
    
    print("\nFine-tuned WavLM Results:")
    for key, value in finetuned_metrics.items():
        print(f"  {key}: {value}")
    
    # Save fine-tuned results
    save_results(
        finetuned_identifications, 
        os.path.join(output_dir, "finetuned_identifications.csv")
    )
    
    # Create confusion matrix for fine-tuned model
    plot_confusion_matrix(
        finetuned_identifications,
        list(set([data['ground_truth_speaker'] for data in finetuned_identifications.values()])),
        os.path.join(output_dir, "finetuned_confusion_matrix.png"),
        "Fine-tuned WavLM Speaker Identification"
    )
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    metrics = ['Rank-1 Accuracy']
    x = np.arange(len(metrics))
    width = 0.35
    
    pretrained_values = [pretrained_metrics[m] for m in metrics]
    finetuned_values = [finetuned_metrics[m] for m in metrics]
    
    plt.bar(x - width/2, pretrained_values, width, label='Pre-trained')
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned')
    
    plt.ylabel('Accuracy')
    plt.title('Speaker Identification Performance')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Add values on bars
    for i, v in enumerate(pretrained_values):
        plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
    
    for i, v in enumerate(finetuned_values):
        plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    
    # Save summary to text file
    with open(os.path.join(output_dir, "identification_summary.txt"), 'w') as f:
        f.write("Speaker Identification on Separated Audio\n")
        f.write("======================================\n\n")
        
        f.write("Pre-trained WavLM Results:\n")
        for key, value in pretrained_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\nFine-tuned WavLM Results:\n")
        for key, value in finetuned_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\nPerformance Difference:\n")
        accuracy_diff = finetuned_metrics['Rank-1 Accuracy'] - pretrained_metrics['Rank-1 Accuracy']
        percentage = (accuracy_diff / pretrained_metrics['Rank-1 Accuracy']) * 100 if pretrained_metrics['Rank-1 Accuracy'] > 0 else float('inf')
        f.write(f"  Accuracy Improvement: {accuracy_diff:.4f} ({percentage:.2f}%)\n")
    
    print("\nEvaluation complete!")
    print(f"Results saved to {output_dir}")
    print(f"Summary saved to {os.path.join(output_dir, 'identification_summary.txt')}")

if __name__ == "__main__":
    main() 