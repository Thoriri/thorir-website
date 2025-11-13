---
title: "BioFoundation: Foundation Models for Biosignals"
summary: "Open-source framework powering LUNA, FEMBA, CEReBrO, and other EEG foundation models — including LUNA, accepted at NeurIPS 2025."
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
url_video: ""

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

BioFoundation is an open-source research framework for developing and deploying foundation models for biomedical signal analysis, with a particular focus on EEG (electroencephalography) data. The project powers the latest generation of topology-agnostic biosignal models — including **LUNA (NeurIPS 2025)**, **FEMBA (EMBC 2025)**, and **CEReBrO (arXiv 2025)** — and provides the shared infrastructure that enables them to scale across datasets, montages, and downstream tasks.

---

## LUNA: Foundation Model for EEG Analysis

**Accepted at NeurIPS 2025** 🏆

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
- **0.921 AUROC** on TUAR artifact-detection benchmark
- Strong transfer across four clinical tasks:
  - Abnormality detection (TUAB)
  - Artifact rejection (TUAR)
  - Slowing classification (TUSL)
  - Emotion recognition (SEED-V)

---

## Flagship Foundation Models

BioFoundation maintains a family of EEG foundation models, all sharing the same training stack, data tooling, and evaluation harness:

### LUNA · NeurIPS 2025
- Topology-agnostic architecture with channel unification and cross-attention
- Linear complexity with respect to electrode count
- Pre-trained on 21k+ hours of EEG and released on Hugging Face: [thorir/LUNA](https://huggingface.co/thorir/LUNA)

### FEMBA · EMBC 2025
- Bidirectional Mamba state-space architecture with linear-time scaling
- 81.82% balanced accuracy on TUAB and 0.949 AUROC on TUAR
- Available on Hugging Face: [thorir/FEMBA](https://huggingface.co/thorir/FEMBA)

### CEReBrO · arXiv 2025
- Alternating-attention encoder that jointly models temporal and spatial correlations
- 2× speed improvement and 6× lower memory vs. dense self-attention
- Released via the BioFoundation codebase (see repository)

All models share:
- **Codebase:** [pulp-bio/biofoundation](https://github.com/pulp-bio/biofoundation)
- **Licensing:** Apache 2.0 for code, CC BY-ND 4.0 for official weight releases
- **Configuration system:** Hydra + PyTorch Lightning for reproducible experiments
- **Evaluation:** Consistent TUAB/TUAR/TUSL benchmarks and downstream EEG tasks

---

## Applications

The BioFoundation models target both clinical and research scenarios:

### Clinical Workflows
- **Abnormality detection** for diagnostic triage
- **Seizure prediction and monitoring** on wearable EEG
- **Artifact rejection** and signal cleaning
- **Sleep staging** and long-term physiological monitoring

### Research & Interfaces
- **Emotion recognition** and affective computing
- **Brain-computer interfaces** with low-channel wearables
- **Cognitive workload and attention tracking**
- **Neurofeedback** with topology-agnostic electrodes

---

## Resources

- ⭐ 29 GitHub stars • 🍴 2 forks (live metrics)
- 📚 **Codebase:** [BioFoundation on GitHub](https://github.com/pulp-bio/biofoundation)
- 🤗 **Model Weights:** [LUNA](https://huggingface.co/thorir/LUNA) · [FEMBA](https://huggingface.co/thorir/FEMBA)
- 📄 **Paper:** [LUNA preprint (arXiv:2510.22257)](https://arxiv.org/abs/2510.22257)
- 🔭 **Roadmap:** Upcoming releases include CEReBrO checkpoints and additional downstream adapters

---

## Citation

If you use BioFoundation in your research, please cite:

```bibtex
@inproceedings{doner2025luna,
  title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
  author={D{\"o}ner, Berkay and Ingolfsson, Thorir Mar and Benini, Luca and Li, Yawei},
  booktitle={Neural Information Processing Systems},
  year={2025}
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

*This project is part of ongoing research at ETH Zurich's Integrated Systems Laboratory (IIS) in collaboration with leading researchers in TinyML and biomedical signal processing.*
