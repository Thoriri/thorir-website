---
title: "Minimizing Artifact-Induced False Alarms for Seizure Detection in Wearable EEG Devices with Gradient-Boosted Tree Classifiers"
publication_short: In *Scientific Reports 14:2980 (2024)*
authors:
  - admin
  - Simone Benatti
  - Xiaying Wang
  - Adriano Bernini
  - Pauline Ducouret
  - Philippe Ryvlin
  - Sándor Beniczky
  - Luca Benini
  - Andrea Cossettini
publication_types:
  - "2"
date: 2024-02-05T00:00:00Z
publishDate: 2024-02-05T00:00:00Z
venue: "Scientific Reports"
abstract: |
  Motion, muscle, and eye-blink artifacts cause false alarms in continuous seizure monitoring. This study proposes a combined seizure and artifact detection scheme using gradient-boosted decision trees tailored for wearable EEG devices with limited channels. On the CHB-MIT dataset, the subject-specific approach yields 65.27% sensitivity and 93.95% artifact-detection accuracy, reducing false alarms by up to 96% compared to standalone seizure detection. An energy-efficient implementation achieves 300-hour battery life on a wearable platform, demonstrating the feasibility of robust, long-term monitoring.
summary: "Gradient-boosted seizure + artifact detection that cuts false alarms by 96% while enabling 300-hour wearable monitoring."
featured: false
projects: []
tags:
  - artifact detection
  - seizure detection
  - gradient boosted trees
  - wearable EEG
url_pdf: "https://www.nature.com/articles/s41598-024-52551-0"
url_arxiv: ""
url_code: ""
url_dataset: "https://physionet.org/content/chbmit/1.0.0/"
url_video: ""
url_slides: ""
url_project: ""
url_doi: "https://doi.org/10.1038/s41598-024-52551-0"
url_poster: ""
image:
  caption: "Seizure and artifact detection workflow"
  focal_point: Smart
  preview_only: false
  filename: featured-1.png
links:
  - icon: external-link-alt
    icon_pack: fas
    name: Scientific Reports Article
    url: "https://www.nature.com/articles/s41598-024-52551-0"
  - icon: database
    icon_pack: fas
    name: CHB-MIT Dataset
    url: "https://physionet.org/content/chbmit/1.0.0/"
---

## Key Highlights

- Combines seizure and artifact detection with gradient-boosted trees to reduce false alarms by up to 96%.
- Achieves 65.27% sensitivity and 93.95% artifact classification accuracy on CHB-MIT and TUH EEG Artifact datasets.
- Optimized embedded implementation enables 300-hour operation on wearable EEG hardware.
