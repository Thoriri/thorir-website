---
title: "BioFoundation: Foundation Models for Biosignals"
summary: "Open-source framework for developing and deploying foundation models for EEG and biosignal analysis, featuring LUNA - an efficient, topology-agnostic EEG foundation model accepted at NeurIPS 2024."
tags:
  - Foundation Models
  - TinyML
  - Biosignals
  - EEG
  - Deep Learning
date: "2024-11-01T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: ""
  focal_point: Smart

links:
  - icon: github
    icon_pack: fab
    name: GitHub
    url: https://github.com/pulp-bio/biofoundation
  - icon: file-pdf
    icon_pack: fas
    name: Paper
    url: https://arxiv.org/abs/2510.22257

url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: "https://arxiv.org/abs/2510.22257"
url_slides: ""
url_video: "https://www.youtube.com/watch?v=1KPFJlJaXTI"

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""

# Featured image
# To use, place an image named `featured.jpg/png` in your page's folder.
# Placement options: 1 = Full column width, 2 = Out-set, 3 = Screen-width
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
# Set `preview_only` to `true` to just use the image for thumbnails.
featured: true
---

## Overview

BioFoundation is an open-source research framework for developing and deploying foundation models for biomedical signal analysis, with a particular focus on EEG (electroencephalography) data. The project hosts our latest work on efficient, topology-agnostic foundation models that can understand diverse biosignals with minimal fine-tuning.

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; margin: 30px 0; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <iframe
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
    src="https://www.youtube.com/embed/1KPFJlJaXTI"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

---

## LUNA: Foundation Model for EEG Analysis

**Accepted at NeurIPS 2024** 🏆

Our flagship model, LUNA (Lightweight Unified Network for EEG Analysis), addresses a critical challenge in brain signal processing: different EEG datasets use varying electrode configurations, which has historically hindered the development of large-scale foundation models.

### Key Innovations

**Topology-Agnostic Architecture**
- Works seamlessly across different electrode layouts
- Uses learned queries and cross-attention mechanisms
- Compresses multi-channel EEG into unified latent representations

**Unprecedented Efficiency**
- **300× reduction in FLOPs** compared to standard transformers
- **10× reduction in GPU memory** usage
- Linear complexity relative to channel count (not quadratic)

**Large-Scale Pretraining**
- Pretrained on **21,000+ hours** of EEG data
- Diverse electrode configurations and recording conditions
- Masked-patch reconstruction objectives

**State-of-the-Art Performance**
- **0.921 AUROC** on TUAR abnormality detection benchmark
- Robust across four clinical tasks:
  - Abnormality detection
  - Artifact rejection
  - Slowing classification
  - Emotion recognition

---

## Technical Details

### Architecture

The BioFoundation framework includes multiple model architectures optimized for different use cases:

**LUNA (NeurIPS 2024)**
- Topology-agnostic design with cross-attention
- Linear-time complexity
- Efficient fine-tuning workflows
- Pre-trained weights available

**FEMBA** (Foundation EEG Model with Bidirectional Mamba)
- Uses linear-time Mamba architecture instead of quadratic attention
- Strong performance on benchmark datasets (TUAB/TUAR/TUSL)
- Memory-efficient training and inference

### Implementation

- **Framework:** PyTorch Lightning for scalable, distributed training
- **Configuration:** Hydra for flexible, reproducible experiments
- **Optimization:** Supports GPU memory optimization through activation checkpointing
- **License:** Apache 2.0 for code, CC BY-ND 4.0 for model weights

---

## Applications

The models in BioFoundation are designed for real-world clinical and research applications:

### Clinical Tasks
- **Abnormality Detection**: Identify pathological patterns in EEG
- **Seizure Prediction**: Early warning for epileptic seizures
- **Artifact Rejection**: Automatic cleaning of corrupted signals
- **Sleep Staging**: Automated sleep phase classification

### Research Applications
- **Emotion Recognition**: Decode affective states from brain signals
- **Brain-Computer Interfaces**: Enable direct neural control
- **Cognitive Monitoring**: Track mental workload and attention
- **Neurofeedback**: Real-time brain state feedback

---

## Getting Started

The repository is structured for easy use and contribution:

```bash
# Clone the repository
git clone https://github.com/pulp-bio/biofoundation.git
cd biofoundation

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
# (Instructions in repository)

# Run inference or fine-tuning
python train.py --config configs/luna_finetune.yaml
```

### Key Features
- Modular architecture (data loading, models, training tasks)
- Pre-trained model weights ready to use
- Example notebooks for common tasks
- Distributed training support
- Comprehensive documentation

---

## Impact & Recognition

- ⭐ **27 stars** on GitHub (growing community)
- 🏆 **Accepted at NeurIPS 2024** (top-tier ML conference)
- 📊 **21,000+ hours** of training data
- 🌍 **Open-source**: Available for research and development

---

## Citation

If you use BioFoundation in your research, please cite:

```bibtex
@article{doner2024luna,
  title={LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis},
  author={Döner, Berkay and Ingolfsson, Thorir Mar and Benini, Luca and Li, Yawei},
  journal={arXiv preprint arXiv:2510.22257},
  year={2024}
}
```

---

## Collaboration

This project is actively developed and we welcome contributions! Whether you're interested in:
- Extending the framework to new biosignals
- Improving model architectures
- Adding new downstream tasks
- Optimizing for edge deployment

**Get involved:**
- 💻 [Contribute on GitHub](https://github.com/pulp-bio/biofoundation)
- 📧 [Contact for collaboration](mailto:thoriri@iis.ee.ethz.ch)
- 📄 [Read the paper](https://arxiv.org/abs/2510.22257)

---

## Related Publications

- Döner, B., **Ingolfsson, T. M.**, Benini, L., & Li, Y. (2024). LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis. *NeurIPS 2024*.

---

*This project is part of ongoing research at ETH Zurich's Integrated Systems Laboratory (IIS) in collaboration with leading researchers in TinyML and biomedical signal processing.*
