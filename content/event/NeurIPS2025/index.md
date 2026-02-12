---
title: "NeurIPS 2025 – LUNA Poster Presentation"

event: "Conference on Neural Information Processing Systems (NeurIPS)"
event_url: "https://neurips.cc/Conferences/2025"
tags:
  - NeurIPS
  - Foundation Models
  - EEG
  - Self-Supervised Learning
  - Topology-Agnostic
location: San Diego Convention Center, San Diego, USA
address:
  street: 111 W Harbor Dr
  city: San Diego
  region: California
  postcode: '92101'
  country: United States

summary: Poster presentation of LUNA—our topology-agnostic EEG foundation model—at NeurIPS 2025 in San Diego.
abstract: |
  Presented our poster on **LUNA (Latent Unified Network Architecture)**, a self-supervised foundation model that reconciles disparate EEG electrode geometries while scaling linearly with channel count. LUNA compresses multi-channel EEG into a fixed-size, topology-agnostic latent space via learned queries and cross-attention, then operates exclusively on this latent representation using patch-wise temporal self-attention—decoupling computation from electrode count. Pre-trained on TUEG and Siena (over 21,000 hours of raw EEG across diverse montages) with a masked-patch reconstruction objective, LUNA transfers effectively to four downstream tasks: abnormality detection, artifact rejection, slowing classification, and emotion recognition. It achieves state-of-the-art results on TUAR and TUSL (e.g., 0.921 AUROC on TUAR), while reducing FLOPs by 300× and GPU memory by up to 10×.

date: "2025-12-04T09:00:00-08:00"
all_day: true

publishDate: "2025-12-01T00:00:00Z"

authors: []

featured: true

image:
  caption: "Presenting the LUNA poster at NeurIPS 2025"
  focal_point: Smart
  filename: LUNA_Poster.jpeg
  preview_only: false

links:
  - icon: file-pdf
    icon_pack: fas
    name: LUNA poster PDF
    url: "uploads/neurips2025_luna_poster.pdf"
  - icon: external-link-alt
    icon_pack: fas
    name: LUNA paper (arXiv)
    url: "https://arxiv.org/abs/2510.22257"
  - icon: github
    icon_pack: fab
    name: BioFoundation codebase
    url: "https://github.com/pulp-bio/biofoundation"
url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: "uploads/neurips2025_luna_poster.pdf"
url_slides: ""
url_video: ""

slides: ""

projects: []
---

## Poster highlights

- **Topology-agnostic design:** LUNA uses learned queries and cross-attention to compress any electrode layout into a fixed-size latent space, enabling training across heterogeneous EEG datasets without channel-specific engineering.
- **Linear scaling:** Patch-wise temporal self-attention operates entirely in the latent space, decoupling compute cost from electrode count—300× fewer FLOPs and 10× less GPU memory than comparable models.
- **Strong transfer:** Pre-trained on over 21,000 hours of TUEG and Siena EEG, LUNA achieves state-of-the-art results on TUAR (0.921 AUROC) and TUSL across abnormality detection, artifact rejection, slowing classification, and emotion recognition.

## Photo gallery

{{< figure src="LUNA_Poster.jpeg" caption="Presenting the LUNA poster at NeurIPS 2025 in San Diego." >}}

## Poster & resources

- [Download the LUNA poster PDF →](uploads/neurips2025_luna_poster.pdf)
- [LUNA paper on arXiv →](https://arxiv.org/abs/2510.22257)
- [BioFoundation GitHub repository →](https://github.com/pulp-bio/biofoundation)

## Related work

- [LUNA publication →](/publication/2025-neurips-luna/)
- [BioFoundation project →](/project/biofoundation/)
