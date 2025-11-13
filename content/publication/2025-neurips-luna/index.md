---
title: "LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis"
publication_short: In *NeurIPS 2025*
authors:
  - Berkay Döner
  - admin
  - Luca Benini
  - Yawei Li
publication_types:
  - "1"
date: 2025-12-01T00:00:00Z
publishDate: 2024-11-13T00:00:00Z
lastmod: 2024-11-13T00:00:00Z
venue: "Conference on Neural Information Processing Systems (NeurIPS)"
abstract: |
  Electroencephalography (EEG) datasets use heterogeneous electrode layouts, which hampers the generalization of large-scale models. LUNA (Latent Unified Network Architecture) is a self-supervised foundation model that compresses multi-channel EEG into a topology-agnostic latent space, allowing patch-wise linear and cross-attention operations that decouple computation from electrode count. Trained on more than 21,000 hours of TUEG and Siena EEG using a masked reconstruction objective, LUNA transfers to tasks such as abnormality detection, artifact rejection, slowing classification, and emotion recognition, achieving state-of-the-art AUROC on TUAR and TUSL while reducing FLOPs by 300× and GPU memory consumption by 10×.
summary: "Topology-agnostic EEG foundation model that delivers state-of-the-art performance with 300× fewer FLOPs and 10× lower memory usage."
featured: true
projects:
  - biofoundation
tags:
  - EEG
  - foundation model
  - topology-agnostic
  - self-supervised learning
  - transformer
url_pdf: "https://arxiv.org/pdf/2510.22257"
url_arxiv: "https://arxiv.org/abs/2510.22257"
url_code: "https://github.com/pulp-bio/biofoundation"
url_dataset: ""
url_video: ""
url_slides: ""
url_project: "https://github.com/pulp-bio/biofoundation"
url_doi: "https://doi.org/10.48550/arXiv.2510.22257"
url_poster: ""
image:
  caption: "LUNA Foundation Model"
  focal_point: Center
  preview_only: false
  filename: featured.png
links:
  - icon: file-pdf
    icon_pack: fas
    name: Preprint (arXiv)
    url: "https://arxiv.org/pdf/2510.22257"
  - icon: code
    icon_pack: fas
    name: GitHub Repository
    url: "https://github.com/pulp-bio/biofoundation"
  - icon: external-link-alt
    icon_pack: fas
    name: DOI
    url: "https://doi.org/10.48550/arXiv.2510.22257"
---

## Key Highlights

- Compresses multi-channel EEG into a topology-agnostic latent space that decouples computation from electrode count.
- Pre-trained on more than 21,000 hours of EEG data and transfers to abnormality detection, artifact rejection, slowing classification, and emotion recognition.
- Achieves 0.921 AUROC on TUAR while reducing FLOPs by 300× and GPU memory requirements by 10×.

## Resources

- 📄 [Preprint on arXiv](https://arxiv.org/abs/2510.22257)
- 💻 [BioFoundation GitHub repository](https://github.com/pulp-bio/biofoundation)
- 🔗 [DOI entry](https://doi.org/10.48550/arXiv.2510.22257)
