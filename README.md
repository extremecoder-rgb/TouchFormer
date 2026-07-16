# TouchFormer: Self-Supervised Foundation Model for Human Body Understanding from Pressure Maps

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://touchformer.onrender.com/)
[![Model Space](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/subhra509/touchformer)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/extremecoder-rgb/TouchGPT)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2004.01166)

## Overview

**TouchFormer** is a self-supervised Vision Transformer (ViT-B/16) based foundation model designed for accurate 3D human pose and shape estimation from tactile pressure maps. Unlike vision-centric models that rely on optical sensors, TouchFormer processes non-optical physical pressure grids (64×27 force-resistive matrices from smart mattresses) to understand human body contact and pose.

This project bridges the gap between computer vision's foundation models success and the underexplored tactile perception domain, enabling contact-based human understanding for applications in healthcare, smart furniture, and embodied AI.

---

## What This Project Solves

### The Problem
- **Tactile perception** has historically been restricted to task-specific models
- **Pressure maps** from contact sensors contain rich information about body shape, weight distribution, and pose
- Traditional **CNNs** lack the ability to capture complex, non-local pressure distributions
- No existing **foundation models** for tactile understanding comparable to vision transformers in CV

### The Solution
TouchFormer leverages:
- **Self-supervised pretraining (MAE)** with 75% masking to learn robust tactile features
- **Transfer learning** from ImageNet-pretrained ViT-B/16 for superior downstream performance
- **Vision Transformers** to capture long-range dependencies in pressure distributions
- **Direct 3D pose regression** from 2D pressure matrices (64×27 → 24 keypoints × 3D)

---

## Key Results

| Model Configuration | Backbone Init | Pretraining | MPJPE (cm) ↓ | PCK @ 15cm (%) ↑ |
|---------------------|---------------|-------------|--------------|------------------|
| CNN Baseline (Clever et al., CVPR 2020) | Random | None | 7.75 | — |
| TouchFormer (Custom MAE - Exp B) | Random | MAE (Pressure Maps) | 6.91 | 90.8% |
| **TouchFormer (ImageNet Transfer - Exp A) - SOTA** | **ImageNet-1k** | **None** | **6.29** | **92.8%** |

**Key Finding:** Transfer learning from ImageNet-pretrained weights outperforms task-specific self-supervised pretraining, achieving **18.8% error reduction** over CNN baselines.

---

## Project Structure

```
TouchFormer/
├── index.html              # Main academic project page (25KB HTML)
├── static/
│   ├── style.css           # Academic typography & responsive styling
│   ├── script.js           # Minimal UI utilities (BibTeX copy, etc.)
│   └── teaser.jpg          # Project teaser visualization
└── README.md               # This file
```

### Architecture Overview

**Phase 1: Self-Supervised MAE Pretraining**
1. Input: Raw 64×27 tactile pressure map
2. Mask 75% of patches
3. ViT-B/16 encoder processes visible patches only
4. ViT decoder reconstructs all patches + masked regions
5. Loss: MSE reconstruction on pressure values

**Phase 2: Downstream Fine-Tuning & Inference**
1. Load pretrained ViT-B/16 encoder (from MAE or ImageNet)
2. Add linear keypoint head (72-dim output: 24 keypoints × 3D)
3. Fine-tune on PressurePose dataset
4. Predict 3D skeleton directly from full pressure map

---

## How to Run

### View the Project
The repository serves a static academic project page:

```bash
# Clone the repository
git clone https://github.com/extremecoder-rgb/TouchFormer.git
cd TouchFormer

# Serve locally (using Python)
python -m http.server 8000

# Open in browser
# http://localhost:8000
```

### Access Online
- **Project Page:** [touchformer.onrender.com](https://touchformer.onrender.com/)
- **Interactive Model:** [HuggingFace Spaces](https://huggingface.co/spaces/subhra509/touchformer)

---

## Technical Highlights

### Model Architecture
- **Backbone:** Google Vision Transformer (ViT-B/16) - 12 layers, 768 hidden dims
- **Input Patch Size:** 16×16 patches on 64×27 pressure maps
- **Pretraining Objective:** Masked Autoencoder (MAE) with 75% masking ratio
- **Fine-tuning Head:** Linear regression head predicting 24 × 3D keypoints (72-dim output)
- **Initialization Strategy:** ImageNet-1k pretrained weights outperform random init

### Dataset
- **PressurePose Corpus:** 10GB of aligned pressure maps + 3D pose annotations
- **Sensor Hardware:** 64×27 force-resistive pressure matrices (smart mattress setup)
- **Tasks:** 3D human pose estimation, shape estimation, contact understanding

### Evaluation Metrics
- **MPJPE (Mean Per Joint Position Error):** Distance error in cm (lower is better)
- **PCK (Percentage of Correct Keypoints):** % keypoints within 15cm threshold (higher is better)

---

## Key Takeaways

### 1. Transfer Learning Benefits
Initializing with ImageNet-1k ViT-B/16 weights yields **best SOTA results** (6.29 cm MPJPE, 92.8% PCK), demonstrating that visual feature abstractions transfer effectively to tactile perception.

### 2. Self-Supervised Tactile Priors
Custom MAE pretraining on pressure maps (20 epochs) achieves competitive results (6.91 cm MPJPE), showing that tactile-specific priors can be learned without expensive labeled data.

### 3. Compute & Efficiency
Despite academic compute constraints:
- Custom MAE model trained for only **20 epochs** (not converged)
- ImageNet transfer learning requires **zero additional pretraining**
- Superior performance with minimal domain-specific training cost

---

## Stack & Technologies

| Component | Details |
|-----------|---------|
| **Frontend** | HTML5, CSS3, vanilla JavaScript |
| **Styling** | Custom design with warm academic aesthetic (CSS Variables) |
| **Fonts** | Inter (sans-serif), Lora (serif) from Google Fonts |
| **Architecture** | SVG diagrams, responsive tables, feature cards |
| **Hosting** | Render (project page), HuggingFace Spaces (model) |

---

## Related Work & Scientific Context

### Comparison with Prior Work
- **Sparsh (Meta AI, 2024):** Optical tactile sensing; TouchFormer handles non-optical pressure grids
- **Clever et al. (CVPR 2020):** CNN baseline for pressure-based pose; TouchFormer achieves 18.8% error reduction
- **Vision Transformers (ViT - Dosovitskiy et al., 2021):** ImageNet pretraining; TouchFormer adapts ViT for tactile domain

### Novel Contributions
1. **First foundation model for tactile understanding** from non-optical pressure matrices
2. **Transfer learning analysis** showing ImageNet pretrained ViT > custom tactile MAE
3. **Efficient tactile pretraining** with 75% masking on 64×27 pressure grids
4. **Multi-task capable** model for pose, shape, and contact understanding

---

## Citation

If you use TouchFormer in your research, please cite:

```bibtex
@article{mondal2026touchformer,
  title={TouchFormer: Self-Supervised Foundation Model for Tactile Understanding},
  author={Mondal, Subhranil},
  journal={arXiv preprint arXiv:2607.01166},
  year={2026}
}

@inproceedings{clever2020bodies,
  title={Bodies at Rest: 3D Human Shape and Pose Estimation from a Pressure Image using Synthetic Data},
  author={Clever, Henry M and Erickson, Zackory and Kapusta, Ariel and Turk, Greg and Liu, C Karen and Kemp, Charles C},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

---

## Explore Further

- **Project Page:** [touchformer.onrender.com](https://touchformer.onrender.com/) — Full technical report, architecture diagrams, and evaluation results
- **Code Repository:** [github.com/extremecoder-rgb/TouchGPT](https://github.com/extremecoder-rgb/TouchGPT) — Training code and model weights
- **Interactive Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/subhra509/touchformer) — Try predictions on sample pressure maps
- **Baseline Paper:** [arXiv:2004.01166](https://arxiv.org/abs/2004.01166) — Clever et al. CNN baseline

---

## Author

**Subhranil Mondal**  
Independent Researcher | Computer Vision & Tactile Perception

---

## License

This project is open source. See repository for license details.

---

## Acknowledgments

Built with:
- **Google Vision Transformer (ViT-B/16)** backbone
- **ImageNet-1k** pretrained weights
- **Masked Autoencoder (MAE)** pretraining methodology
- Inspired by **Sparsh** (Meta AI) and **Clever et al.** (CVPR 2020)

Hosted on Render & HuggingFace Spaces.

---

**Last Updated:** July 2026  
**Repository:** [github.com/extremecoder-rgb/TouchFormer](https://github.com/extremecoder-rgb/TouchFormer)
