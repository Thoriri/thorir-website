---
title: "IEEE EMBC 2025 – Silent Disco Presentations"

event: "IEEE Engineering in Medicine & Biology Conference (EMBC)"
event_url: "https://embc.embs.org/2025/"
tags:
  - EMBC
  - Foundation Models
  - Biosignals
  - Quantization
location: Bella Center Copenhagen, Denmark
address:
  street: Center Blvd. 5
  city: Copenhagen
  region: Capital Region
  postcode: '2300'
  country: Denmark

summary: Presented quantized biosignal foundation models in EMBC’s silent disco session while Anna Tegon delivered FEMBA.
abstract: |
  EMBC 2025 experimented with “silent disco” technical sessions—headphones on, presentations streamed like a live podcast. I introduced our finetuning and quantisation pipeline for EEG foundation models on ECG/PPG blood-pressure estimation, while Anna Tegon showcased FEMBA. Together we highlighted how BioFoundation unifies multi-modality biosignal learning.

date: "2025-07-17T11:00:00+02:00"
all_day: false

publishDate: "2025-07-10T00:00:00Z"

authors: []

featured: true

image:
  caption: "Silent-disco style oral at IEEE EMBC 2025"
  focal_point: Smart
  filename: EMBC_talk.jpeg
  preview_only: false

links:
  - icon: file-pdf
    icon_pack: fas
    name: Blood-pressure slides
    url: "uploads/embc2025_bp_slides.pdf"
  - icon: file-pdf
    icon_pack: fas
    name: FEMBA slides
    url: "uploads/embc2025_femba_slides.pdf"
  - icon: external-link-alt
    icon_pack: fas
    name: Finetuning & Quantization paper
    url: "https://arxiv.org/abs/2502.17460"
  - icon: external-link-alt
    icon_pack: fas
    name: FEMBA paper
    url: "https://arxiv.org/abs/2502.06438"
  - icon: github
    icon_pack: fab
    name: BioFoundation codebase
    url: "https://github.com/pulp-bio/biofoundation"
url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: ""
url_slides: "uploads/embc2025_bp_slides.pdf"
url_video: ""

slides: ""

projects: []
---

## Presentation highlights

- **Silent disco format:** Attendees tuned into talks through headsets—an unexpectedly intimate way to deliver technical content.
- **Finetuning & Quantization talk:** Covered cross-modality transfer from EEG foundation models to ECG/PPG and INT8/INT4 deployment on edge devices.
- **FEMBA spotlight:** Anna Tegon (ETH Zürich) presented our linear-time Mamba architecture for biosignals.
- **Takeaway:** Foundation models for biosignals are converging—cross-modality pretraining, efficient finetuning, and open tooling via BioFoundation.

## Photo gallery

{{< figure src="EMBC_talk.jpeg" caption="Headphones on for the silent-disco oral session." >}}

{{< figure src="EMBC_anna_thorir.jpeg" caption="Catching up with Anna Tegon after the FEMBA presentation." >}}

## Slides & resources

- [Download the blood-pressure slide deck →](uploads/embc2025_bp_slides.pdf)
- [Download the FEMBA slide deck →](uploads/embc2025_femba_slides.pdf)
- Paper links are already live; code and checkpoints will land in the BioFoundation repository.

## Related work

- [Blood-pressure finetuning paper →](https://arxiv.org/abs/2502.17460)
- [FEMBA paper →](https://arxiv.org/abs/2502.06438)
- [BioFoundation project →](/project/biofoundation/)

