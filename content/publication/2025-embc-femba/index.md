---
title: "FEMBA: Efficient and Scalable EEG Analysis with a Bidirectional Mamba Foundation Model"
publication_short: In *EMBC 2025*
authors:
  - Anna Tegon
  - admin
  - Xiaying Wang
  - Luca Benini
  - Yawei Li
publication_types:
  - "1"
date: 2025-07-01T00:00:00Z
publishDate: 2025-07-01T00:00:00Z
venue: "IEEE Engineering in Medicine & Biology Conference (EMBC)"
abstract: |
  FEMBA proposes a self-supervised EEG foundation model built on a bidirectional Mamba state-space architecture that scales linearly with sequence length and avoids the quadratic complexity of transformers. Trained on more than 21,000 hours of unlabeled EEG and fine-tuned on multiple downstream tasks, FEMBA reaches 81.82% balanced accuracy (AUROC 0.8921) on the TUAB dataset and 0.949 AUROC on TUAR, while a tiny 7.8M-parameter variant demonstrates suitability for resource-constrained devices.
summary: "Bidirectional Mamba EEG foundation model that matches transformer accuracy while scaling linearly with sequence length."
featured: true
projects:
  - biofoundation
tags:
  - EEG
  - foundation model
  - mamba
  - self-supervised learning
  - state-space model
url_pdf: "https://arxiv.org/pdf/2502.06438"
url_arxiv: "https://arxiv.org/abs/2502.06438"
url_code: "https://github.com/pulp-bio/biofoundation"
url_dataset: ""
url_video: ""
url_slides: ""
url_project: "https://github.com/pulp-bio/biofoundation"
url_doi: "https://doi.org/10.48550/arXiv.2502.06438"
url_poster: ""
image:
  caption: "FEMBA Foundation Model"
  focal_point: Center
  preview_only: false
  filename: featured.png
links:
  - icon: file-pdf
    icon_pack: fas
    name: Preprint (arXiv)
    url: "https://arxiv.org/pdf/2502.06438"
  - icon: code
    icon_pack: fas
    name: GitHub Repository
    url: "https://github.com/pulp-bio/biofoundation"
  - icon: external-link-alt
    icon_pack: fas
    name: DOI
    url: "https://doi.org/10.48550/arXiv.2502.06438"
---

## Key Highlights

- Bidirectional Mamba state-space architecture scales linearly with sequence length and avoids transformer quadratic complexity.
- Pre-trained on more than 21,000 hours of EEG data; achieves 81.82% balanced accuracy on TUAB and 0.949 AUROC on TUAR.
- Compact 7.8M-parameter variant is suitable for embedded devices and edge deployment.

## Resources

- 📄 [Preprint on arXiv](https://arxiv.org/abs/2502.06438)
- 💻 [BioFoundation GitHub repository](https://github.com/pulp-bio/biofoundation)
- 🔗 [DOI entry](https://doi.org/10.48550/arXiv.2502.06438)
