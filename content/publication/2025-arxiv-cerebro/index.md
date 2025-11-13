---
title: "CEReBrO: Compact Encoder for Representations of Brain Oscillations Using Efficient Alternating Attention"
publication_short: arXiv:2501.10885 (2025)
authors:
  - Alexandru Dimofte
  - Glenn Anta Bucagu
  - admin
  - Xiaying Wang
  - Andrea Cossettini
  - Luca Benini
  - Yawei Li
publication_types:
  - "3"
date: 2025-01-25T00:00:00Z
publishDate: 2025-01-25T00:00:00Z
venue: "arXiv Preprint"
abstract: |
  CEReBrO introduces a compact EEG foundation model that tokenizes multi-channel recordings into per-channel patches and employs an alternating attention mechanism to jointly model intra-channel temporal dynamics and inter-channel spatial correlations. Compared to standard self-attention, the architecture delivers a two-fold speedup and requires six times less memory while supporting models with 3.6–85 million parameters. Pre-trained on more than 20,000 hours of publicly available EEG data with diverse channel configurations, CEReBrO sets new benchmarks for emotion and seizure detection tasks and maintains competitive performance for anomaly classification and gait prediction.
summary: "Alternating-attention EEG foundation model pre-trained on 20k+ hours that doubles speed and cuts memory by 6× versus standard transformers."
featured: false
projects:
  - biofoundation
tags:
  - EEG
  - foundation model
  - alternating attention
  - self-supervised learning
url_pdf: "https://arxiv.org/pdf/2501.10885"
url_arxiv: "https://arxiv.org/abs/2501.10885"
url_code: ""
url_dataset: ""
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.48550/arXiv.2501.10885"
url_poster: ""
links:
  - icon: file-pdf
    icon_pack: fas
    name: Preprint (arXiv)
    url: "https://arxiv.org/pdf/2501.10885"
  - icon: external-link-alt
    icon_pack: fas
    name: DOI
    url: "https://doi.org/10.48550/arXiv.2501.10885"
---

## Key Highlights

- Alternating attention tokenizes EEG into per-channel patches and jointly captures temporal and spatial correlations.
- Achieves 2× speed improvement and 6× lower memory usage compared to standard self-attention architectures.
- Pre-trained on over 20,000 hours of EEG, setting new benchmarks for emotion and seizure detection tasks.
