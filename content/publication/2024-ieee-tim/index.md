---
title: "A Muscle Pennation Angle Estimation Framework From Raw Ultrasound Data for Wearable Biomedical Instrumentation"
publication_short: In *IEEE TIM 2024*
authors:
  - Sergei Vostrikov
  - admin
  - Soley Hafthorsdottir
  - Christoph Leitner
  - Michele Magno
  - Luca Benini
  - Andrea Cossettini
publication_types:
  - "2"
date: 2024-03-25T00:00:00Z
publishDate: 2024-03-25T00:00:00Z
venue: "IEEE Transactions on Instrumentation and Measurement"
abstract: |
  Muscle pennation angles are key biomarkers for musculoskeletal research and rehabilitation. This work introduces a framework that estimates pennation angles directly from raw 32-channel ultrasound data using feature extraction and an XGBoost regressor aligned with automatic annotations. The method delivers near real-time predictions with a root-mean-square error of 1.6° while compressing the model footprint to 11 kB, enabling execution on a GAP9 microcontroller with 1.31 ms latency and 1.03 mJ energy consumption. The approach unlocks wearable biomedical instrumentation for continuous muscle analysis.
summary: "Ultrasound-based pennation estimation with 1.6° RMSE and 11 kB model running in 1.31 ms on GAP9."
featured: false
projects: []
tags:
  - ultrasound
  - pennation angle
  - XGBoost
  - wearable
  - GAP9
url_pdf: ""
url_arxiv: ""
url_code: ""
url_dataset: ""
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.1109/TIM.2023.3335535"
url_poster: ""
image:
  caption: "Wearable ultrasound pennation analysis"
  focal_point: Smart
  preview_only: false
  filename: featured.PNG
links:
  - icon: external-link-alt
    icon_pack: fas
    name: IEEE Xplore
    url: "https://doi.org/10.1109/TIM.2023.3335535"
---

## Key Highlights

- Estimates muscle pennation angles directly from raw 32-channel ultrasound using an XGBoost regressor with automatic annotations.
- Achieves ~1.6° RMSE with an 11 kB model, 1.31 ms inference latency, and 1.03 mJ energy per prediction on GAP9.
- Enables wearable biomedical instrumentation for continuous muscle analysis and rehabilitation monitoring.