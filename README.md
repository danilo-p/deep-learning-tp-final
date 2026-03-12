# Crack Segmentation with U-Net and ResNet Encoders

A deep learning research project for **semantic segmentation of surface cracks** using U-Net architectures with pretrained ResNet encoders. This work systematically evaluates multiple backbone architectures and training strategies to identify optimal configurations for crack detection.

## Overview

Automated crack detection is critical for infrastructure inspection and maintenance. This project trains and benchmarks U-Net models with ResNet encoders (18, 34, 50 layers) on a large-scale crack segmentation dataset, investigating the impact of:

- Encoder architecture depth (ResNet18 vs 34 vs 50)
- Differential learning rates for encoder vs. decoder
- Encoder freezing during early training
- Tversky loss hyperparameter tuning

**Best result: Dice Score of 0.7429** with a ResNet50 backbone and differential learning rates.

## Dataset

- **Source:** [Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset) (Kaggle)
- **Train:** 9,603 image-mask pairs
- **Test:** 1,695 image-mask pairs
- **Format:** RGB images with binary pixel-level masks (0 = no crack, 255 = crack)
- **Preprocessing:** Resized to 448×448 pixels

## Architecture

**Base Model:** U-Net (encoder-decoder) via `segmentation_models_pytorch`

| Component | Details |
|-----------|---------|
| Encoders | ResNet18, ResNet34, ResNet50 (ImageNet pretrained) |
| Decoder | U-Net standard decoder |
| Output | Single-channel binary mask |
| Loss | Tversky Loss (α=0.5, β=0.5 by default) |
| Optimizer | Adam with per-component learning rates |
| Precision | Mixed precision (AMP + GradScaler) |

## Experimental Results

15 configurations were evaluated across 3 architectures (5 per model):

| Model | Configuration | Dice Score |
|-------|--------------|-----------|
| **ResNet50** | LR_Enc=5e-5, LR_Dec=5e-4 | **0.7429** |
| ResNet34 | LR_Enc=5e-5, LR_Dec=5e-4 | 0.7390 |
| ResNet18 | LR_Enc=5e-5, LR_Dec=5e-3 | 0.7328 |
| ResNet34 | Encoder frozen 5 epochs | 0.7244 |
| ResNet18 | Baseline (equal LR) | 0.7207 |
| ResNet50 | Baseline (equal LR) | 0.7082 |
| ResNet34 | Baseline (equal LR) | 0.6022 |
| ResNet50 | Encoder frozen 5 epochs | 0.5245 |

**Summary statistics:** Mean 0.6995 · Median 0.7148 · Std 0.0485 · Range 0.5245–0.7429

## Key Findings

### 1. Differential Learning Rates Are Critical
Using a lower learning rate for the pretrained encoder (5e-5) and a higher rate for the decoder (5e-4 to 5e-3) consistently outperformed equal-rate configurations by 3–5%. This preserves pretrained ImageNet features while allowing the decoder to rapidly adapt to the segmentation task.

### 2. Encoder Freezing Is Architecture-Dependent
- **ResNet18:** Hurt performance (0.6976 vs 0.7207 baseline)
- **ResNet34:** Improved performance significantly (0.7244 vs 0.6022 baseline, +20%)
- **ResNet50:** Severely degraded performance (0.5245 vs 0.7082 baseline)

No universal rule applies — the effectiveness of freezing depends heavily on the model's capacity and the learning dynamics of each architecture.

### 3. Balanced Dice Loss Outperforms Tuned Tversky
Shifting Tversky weights to penalize false positives more (α=0.55, β=0.45) consistently underperformed balanced settings (α=β=0.5). For this dataset, a standard Dice loss is the best choice.

### 4. ResNet34 Is the Practical Sweet Spot
ResNet34 achieves 0.7390 Dice — only 0.39% below ResNet50's best (0.7429) — while being ~33% lighter. For production or resource-constrained environments, ResNet34 offers the best accuracy-to-compute ratio.

## Training Details

**Data augmentation (training only):**
- Random horizontal and vertical flips (p=0.5)
- Random 90°/180°/270° rotations (p=0.5)
- Color jitter — brightness and contrast (p=0.8)
- ImageNet normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])

**Training setup:**
- Early stopping: patience=10 on test Dice score
- Batch sizes: 128 (ResNet18/34/50), 64 (ResNet101), 32 (ResNet152)
- All seeds fixed to 42 for full reproducibility

**Reproducibility:**
```python
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```

## Dependencies

```
torch>=2.9.0 (CUDA 12.6)
torchvision>=0.24.0
segmentation_models_pytorch==0.5.0
timm==1.0.22
opencv-python
numpy==2.0.2
Pillow==11.3.0
matplotlib
tqdm
```

The project was developed and run on **Google Colab** with GPU access. Dataset access requires the Kaggle API; checkpoints are persisted to Google Drive.

## Repository Structure

```
deep-learning-tp-final/
├── deep_learning_tp_final_crack_segmentation.ipynb   # Main notebook (all experiments)
└── res-u-crack50-semantic-crack-segmentation.pdf      # Final research report
```

## Evaluation Metric

**Dice Coefficient** (F1 score for segmentation):

```
Dice = 2×TP / (2×TP + FP + FN)
```

Computed per image and averaged across the 1,695 test images. Ranges from 0 (no overlap) to 1 (perfect segmentation).

## Recommended Configuration

For best accuracy:
- **Encoder:** ResNet50 (ImageNet pretrained)
- **Decoder:** Standard U-Net
- **Learning rates:** Encoder=5e-5, Decoder=5e-4
- **Loss:** Tversky with α=β=0.5
- **Batch size:** 128

For best accuracy/compute tradeoff, use **ResNet34** with the same learning rate settings (Dice=0.7390, ~33% fewer parameters than ResNet50).
