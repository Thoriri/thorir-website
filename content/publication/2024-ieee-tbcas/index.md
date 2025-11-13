---
title: "Reducing False Alarms in Wearable Seizure Detection With EEGformer: A Compact Transformer Model for MCUs"
publication_short: In *IEEE TBioCAS 2024*
authors:
  - Paola Busia
  - Andrea Cossettini
  - admin
  - Simone Benatti
  - Alessio Burrello
  - Victor J. B. Jung
  - Moritz Scherer
  - Matteo A. Scrugli
  - Adriano Bernini
  - Pauline Ducouret
  - Philippe Ryvlin
  - Paolo Meloni
  - Luca Benini
publication_types:
  - "2"
date: 2024-03-25T00:00:00Z
publishDate: 2024-03-25T00:00:00Z
venue: "IEEE Transactions on Biomedical Circuits and Systems"
abstract: |
  Long-term seizure monitoring in wearables suffers from high false-alarm rates. EEGformer is a compact transformer-based detector tailored for low-power microcontroller units that operates directly on raw temporal EEG channels. Hardware-aware optimization enables EEGformer to detect 73% of seizures on the CHB-MIT dataset with only 0.15 false positives per hour while reducing detection latency by 20%. Deployment on the GAP9 MCU performs inference in 13.7 ms at 0.31 mJ per inference, demonstrating practical suitability for wearable seizure detection devices with multi-day autonomy.
summary: "Compact transformer seizure detector that achieves 73% sensitivity with 0.15 FP/h and runs in 13.7 ms at 0.31 mJ on GAP9."
featured: false
projects:
  - EEGformer
tags:
  - EEGformer
  - seizure detection
  - transformer
  - low-power MCU
url_pdf: "https://ieeexplore.ieee.org/abstract/document/10412626"
url_arxiv: ""
url_code: ""
url_dataset: "https://physionet.org/content/chbmit/1.0.0/"
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.1109/TBCAS.2024.3357509"
url_poster: ""
image:
  caption: "EEGformer hardware-aware deployment"
  focal_point: Smart
  preview_only: false
  filename: featured.png
links:
  - icon: external-link-alt
    icon_pack: fas
    name: IEEE Xplore
    url: "https://ieeexplore.ieee.org/document/10412626"
  - icon: database
    icon_pack: fas
    name: CHB-MIT Dataset
    url: "https://physionet.org/content/chbmit/1.0.0/"
---

## Key Highlights

- Tiny transformer architecture tailored for raw temporal EEG and low-channel wearable acquisition.
- Detects 73% of seizures with 0.15 false positives per hour and reduces detection latency by 20% on CHB-MIT.
- Hardware-aware implementation achieves 13.7 ms inference latency at 0.31 mJ on GAP9, enabling multi-day wearable deployment.