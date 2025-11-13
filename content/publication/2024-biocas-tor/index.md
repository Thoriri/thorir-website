---
title: "Train-On-Request: An On-Device Continual Learning Workflow for Adaptive Real-World Brain-Machine Interfaces"
publication_short: In *IEEE BioCAS 2024*
authors:
  - Lan Mei
  - Cristian Cioflan
  - admin
  - Victor Kartsch
  - Andrea Cossettini
  - Xiaying Wang
  - Luca Benini
publication_types:
  - "1"
date: 2024-10-01T00:00:00Z
publishDate: 2024-10-01T00:00:00Z
venue: "IEEE Biomedical Circuits and Systems Conference (BioCAS)"
abstract: |
  Train-On-Request (TOR) introduces an on-device continual learning workflow that allows users to update brain-machine interface models on demand, addressing inter-session variability while maintaining high accuracy in real-world settings. Evaluated on a motor-movement dataset collected with a wearable headband, TOR achieves up to 92% accuracy with calibration times as low as 1.6 minutes, reducing calibration effort by 46% compared to naive retraining. On a GAP9 RISC-V SoC, on-device training runs in 21.6 ms per step and consumes about 1 mJ, demonstrating feasibility for ultra-low-power edge devices.
summary: "On-device continual learning for wearable BMIs that reaches 92% accuracy with 1.6 min recalibration and 1 mJ training steps."
featured: false
projects:
  - BioGAP TOR
tags:
  - continual learning
  - brain-machine interface
  - on-device training
  - ultra-low power
url_pdf: "https://arxiv.org/pdf/2409.09161"
url_arxiv: "https://arxiv.org/abs/2409.09161"
url_code: "https://github.com/pulp-bio/bmi-odcl"
url_dataset: ""
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.48550/arXiv.2409.09161"
url_poster: ""
links:
  - icon: file-pdf
    icon_pack: fas
    name: Preprint (arXiv)
    url: "https://arxiv.org/pdf/2409.09161"
  - icon: code
    icon_pack: fas
    name: GitHub Repository
    url: "https://github.com/pulp-bio/bmi-odcl"
  - icon: external-link-alt
    icon_pack: fas
    name: DOI
    url: "https://doi.org/10.48550/arXiv.2409.09161"
---

## Key Highlights

- Enables user-initiated on-device continual learning, cutting recalibration time by 46% while sustaining up to 92% accuracy.
- Runs on GAP9 with 21.6 ms training steps consuming about 1 mJ per update, suitable for battery-powered wearables.
- Demonstrates a practical workflow for adaptive BMIs deployed in real-world environments.
