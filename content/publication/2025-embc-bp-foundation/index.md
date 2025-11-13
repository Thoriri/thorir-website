---
title: "Finetuning and Quantization of EEG-Based Foundational BioSignal Models on ECG and PPG Data for Blood Pressure Estimation"
publication_short: In *EMBC 2025*
authors:
  - Bálint Tóth
  - Dominik Senti
  - admin
  - Jeffrey Zweidler
  - Alexandre Elsig
  - Luca Benini
  - Yawei Li
publication_types:
  - "1"
date: 2025-07-01T00:00:00Z
publishDate: 2025-07-01T00:00:00Z
venue: "IEEE Engineering in Medicine & Biology Conference (EMBC)"
abstract: |
  This work explores whether EEG-pretrained foundation models can transfer to other biosignals for cuffless blood-pressure estimation. An EEG foundation model is fine-tuned on photoplethysmography (PPG) and electrocardiography (ECG) signals from the MIMIC-III and VitalDB datasets, achieving near state-of-the-art diastolic blood-pressure estimation (mean absolute error ≈ 1.57 mmHg) and surpassing prior methods for systolic BP with a mean absolute error of 2.72 mmHg. Dynamic INT8 quantization compresses the smallest model from 13.73 MB to 3.83 MB without loss of accuracy, demonstrating feasibility for deployment on constrained devices.
summary: "Transfers EEG foundation models to ECG/PPG for BP estimation, reaching 1.57 mmHg MAE with 3.5× model compression via INT8 quantization."
featured: false
projects:
  - biofoundation
tags:
  - blood pressure estimation
  - foundation models
  - transfer learning
  - ECG
  - PPG
  - quantization
url_pdf: "https://arxiv.org/pdf/2502.17460"
url_arxiv: "https://arxiv.org/abs/2502.17460"
url_code: ""
url_dataset: ""
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.48550/arXiv.2502.17460"
url_poster: ""
links:
  - icon: file-pdf
    icon_pack: fas
    name: Preprint (arXiv)
    url: "https://arxiv.org/pdf/2502.17460"
  - icon: external-link-alt
    icon_pack: fas
    name: DOI
    url: "https://doi.org/10.48550/arXiv.2502.17460"
---

## Key Highlights

- Demonstrates transfer of EEG-based foundation models to ECG/PPG for cuffless blood-pressure estimation.
- Achieves 1.57 mmHg MAE on diastolic BP and 2.72 mmHg on systolic BP across MIMIC-III and VitalDB.
- Dynamic INT8 quantization reduces model size from 13.73 MB to 3.83 MB with negligible accuracy loss.
