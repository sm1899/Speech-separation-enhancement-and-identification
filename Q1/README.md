# Speaker-Aware Speech Separation

A novel end-to-end architecture that combines speech separation with speaker identification in a GAN-inspired framework. This system jointly optimizes speech separation and speaker identification using parameter-efficient fine-tuning of pre-trained models.

## Overview

This project implements an innovative approach to multi-speaker speech separation with two key components:
- A SepFormer-based generator for speech source separation
- A WavLM-based discriminator for speaker identification

Unlike traditional pipelines that process these tasks sequentially, our architecture leverages speaker identification capabilities to guide the separation process, creating more speaker-aware separations.

## Features

- **Joint Optimization**: Speech separation and speaker identification are jointly optimized
- **Parameter-Efficient**: Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Multi-Component Loss**: Combines separation, perceptual, and adversarial losses
- **Multi-GPU Support**: Distributes computation across multiple GPUs for memory efficiency
- **Comprehensive Evaluation**: Measures both separation quality and speaker identification accuracy

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU(s) with at least 24GB combined memory

### Setup

1. Clone the repository:
```bash
git clone https://github.com/username/speaker-aware-separation.git
cd speaker-aware-separation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (automatic on first run) or manually:
```bash
# SepFormer model from SpeechBrain will be downloaded automatically
# For WavLM
python -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-base-plus')"
```

## Dataset Preparation

### Creating a Mixed Speaker Dataset

Use the `vox2mix.py` script to create a dataset of mixed utterances from VoxCeleb2:

```bash
python vox2mix.py --vox2_path /path/to/voxceleb2 --output_dir vox2mix_dataset
```

The script will:
- Select the first 50 speakers for training and the next 50 for testing
- Create 500 training mixtures and 200 test mixtures
- Apply random SNR values and overlap ratios
- Save mixtures, source files, and metadata

## Training

### Fine-tuning the Combined Model

To train the model with default parameters:

```bash
python finetune_combined.py --data_dir vox2mix_dataset --results_dir combined_model_results
```

Advanced options:
```bash
python finetune_combined.py \
  --data_dir vox2mix_dataset \
  --results_dir combined_model_results \
  --batch_size 1 \
  --num_epochs 30 \
  --lr_g 1e-5 \
  --lr_d 1e-3 \
  --sepformer_lora_rank 32 \
  --use_amp \
  --effective_batch_size 16 \
  --max_duration 8.0
```

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir` | Directory with mixed utterances | `vox2mix_dataset` |
| `--results_dir` | Directory to save results | `combined_model_lora_results` |
| `--batch_size` | Batch size | 1 |
| `--num_epochs` | Number of training epochs | 30 |
| `--lr_g` | Generator learning rate | 1e-5 |
| `--lr_d` | Discriminator learning rate | 1e-3 |
| `--sepformer_lora_rank` | LoRA rank for SepFormer | 32 |
| `--use_amp` | Enable mixed precision training | False |
| `--effective_batch_size` | Effective batch size (gradient accumulation) | 16000 |
| `--max_duration` | Maximum audio duration in seconds | 8.0 |

## Evaluation

Evaluate the model on the test set:

```bash
python evaluate_combined.py \
  --data_dir vox2mix_dataset \
  --model_dir combined_model_lora_results \
  --output_dir separated_combined
```

This will:
- Process all test mixtures
- Save separated audio files
- Calculate separation metrics (SDR, SIR, SAR, PESQ)
- Generate evaluation plots

## Comparison with Baseline

To compare with the baseline SepFormer approach:

```bash
# First, separate using the baseline SepFormer
python separation.py

# Then evaluate speaker identification on baseline separation
python evaluate_separation.py

# Finally, compare with our approach
python compare_results.py \
  --baseline_dir separated_vox2mix \
  --proposed_dir separated_combined
```

## Project Structure

```
├── finetune_combined.py      # Main script for training the combined model
├── evaluate_combined.py      # Evaluation of the combined model
├── vox2mix.py                # Dataset creation script
├── separation.py             # Baseline SepFormer separation script
├── evaluate_separation.py    # Speaker identification after separation script
├── src/
│   ├── data_utils.py         # Dataset utilities
│   ├── model_utils.py        # Model loading and adaptation utilities
│   ├── separation_utils.py   # Speech separation utilities
│   ├── sepformer_lora.py     # LoRA adaptation for SepFormer
│   ├── metrics.py            # Evaluation metrics
│   └── loss_utils.py         # Loss functions
├── models/                   # Saved model weights (created during training)
├── vox2mix_dataset/          # Dataset of mixed utterances (created by vox2mix.py)
│   ├── train/
│   │   ├── mixtures/         # Mixed utterances
│   │   ├── sources/          # Source utterances
│   │   └── metadata.csv      # Training metadata
│   └── test/
│       ├── mixtures/         # Mixed utterances
│       ├── sources/          # Source utterances
│       └── metadata.csv      # Test metadata
└── separated_combined/       # Separation results (created during evaluation)
    ├── separated/            # Separated audio files
    ├── test_results.csv      # Detailed results for each test sample
    ├── metrics_summary.txt   # Summary of separation metrics
    └── metrics_distribution.png # Distribution plots of metrics
```


## Key Technologies

- **SepFormer**: Transformer-based speech separation model from SpeechBrain
- **WavLM**: Self-supervised speech representation model from Microsoft
- **LoRA**: Parameter-efficient fine-tuning technique
- **GAN-inspired Architecture**: Using discriminator feedback to improve generator

## Limitations

- Requires substantial GPU memory (2 * 24GB+ recommended)
- Training is computationally intensive
- Trade-off between separation quality and speaker characteristic preservation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{SpeakerAwareSeparation2023,
  author = {Speaker Recognition Team},
  title = {Speaker-Aware Speech Separation with GAN-Inspired Architecture},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/username/speaker-aware-separation}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the WavLM model
- SpeechBrain team for the SepFormer model
- VoxCeleb team for the dataset 