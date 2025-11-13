---
title: "BISeizuRe: BERT-Inspired Seizure Data Representation to Improve Epilepsy Monitoring"
publication_short: In *IEEE EMBC 2024*
authors:
  - Luca Benfenati
  - admin
  - Andrea Cossettini
  - Daniele Jahier Pagliari
  - Alessio Burrello
  - Luca Benini
publication_types:
  - "1"
date: 2024-09-01T00:00:00Z
publishDate: 2024-09-01T00:00:00Z
venue: "IEEE Engineering in Medicine & Biology Conference (EMBC)"
abstract: |
  BISeizuRe leverages a BERT-inspired encoder (BENDR) for EEG-based seizure detection. The model follows a two-phase training strategy—pre-training on the large Temple University Hospital EEG corpus to learn general EEG representations and fine-tuning on CHB-MIT. Subject-specific fine-tuning reduces false positives per hour to 0.23 FP/h, 2.5× lower than the baseline, while maintaining competitive sensitivity. The study analyses architecture choices, pre-processing, and post-processing pipelines to deliver robust seizure detection for wearable monitoring.
summary: "BERT-inspired EEG encoder that cuts seizure-detection false positives to 0.23 FP/h after subject-specific tuning."
featured: false
projects:
  - biofoundation
tags:
  - seizure detection
  - EEG
  - foundation model
  - fine-tuning
url_pdf: "https://arxiv.org/pdf/2406.19189"
url_arxiv: "https://arxiv.org/abs/2406.19189"
url_code: ""
url_dataset: "https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml"
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.48550/arXiv.2406.19189"
url_poster: ""
links:
  - icon: file-pdf
    icon_pack: fas
    name: Preprint (arXiv)
    url: "https://arxiv.org/pdf/2406.19189"
  - icon: database
    icon_pack: fas
    name: TUH EEG Corpus
    url: "https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml"
---

## Key Highlights

- Uses a BERT-inspired encoder (BENDR) to learn generalized EEG representations from large unlabeled data.
- Subject-specific fine-tuning reduces false positives to 0.23 FP/h—2.5× lower than the baseline—while keeping sensitivity high.
- Explores architecture, pre-processing, and post-processing design choices tailored for wearable seizure monitoring.
