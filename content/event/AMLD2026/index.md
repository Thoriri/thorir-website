---
title: "AMLD Intelligence Summit 2026 – Pushing Biosignal FMs to the Edge"

event: "AMLD Intelligence Summit 2026 (Applied Machine Learning Days)"
event_url: "https://2026.appliedmldays.org/"
tags:
  - AMLD
  - Foundation Models
  - Biosignals
  - Edge Deployment
  - Quantization
  - TinyML
location: SwissTech Convention Center, EPFL, Lausanne, Switzerland
address:
  street: Route Louis-Favre 2
  city: Ecublens
  region: Vaud
  postcode: '1024'
  country: Switzerland

summary: Poster presentation of our full-stack framework for deploying biosignal foundation models on ultra-low-power edge devices at AMLD 2026 in Lausanne.
abstract: |
  Presented our poster on **Pushing Biosignal Foundation Models to Ultra-low-power Edge Devices**, a unified full-stack framework for deploying EEG and EMG foundation models on the GreenWaves Technologies GAP9 ultra-low-power RISC-V microcontroller. We showcased the deployment path for four architectures: **CEReBrO**—a compact encoder using alternating attention that achieves the fastest inference at 146 ms and just 8 mJ per window; **FEMBA**—a bidirectional Mamba FM quantized down to 2-bit weights using QAT; **TinyMyo**—a 3.6 M-parameter Vision Transformer for gestures, kinematics, and silent speech running at ~36 mW always-on; and **LUNA**—our topology-agnostic EEG foundation model, now deployed on-device with channel unification via RoPE, FFT, and learned queries. The pipeline spans PyTorch training through Brevitas quantization-aware training, INT8 export, custom parallel C kernel generation, and hierarchical L3→L2→L1 weight streaming with double-buffered DMA on GAP9. Quantization compresses models up to 9× (FP32→INT8) with less than 2% accuracy drop, and the neural accelerator (NE16) boosts throughput to over 10 MACs/cycle—enabling the first real-time deployment of state-of-the-art biosignal foundation models on an ultra-low-power edge device.

date: "2026-02-10T09:00:00+01:00"
all_day: true

publishDate: "2026-02-01T00:00:00Z"

authors: []

featured: true

image:
  caption: "Presenting the FM edge-deployment poster at AMLD 2026"
  focal_point: Smart
  filename: FM_Ultra_edge.jpeg
  preview_only: false

links:
  - icon: external-link-alt
    icon_pack: fas
    name: AMLD 2026 event page
    url: "https://2026.appliedmldays.org/"
url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: "uploads/amld2026_fm_edge_poster.pdf"
url_slides: ""
url_video: ""

slides: ""

projects: []
---

## Poster highlights

- **Four foundation models, one pipeline:** CEReBrO, FEMBA, LUNA, and TinyMyo—spanning EEG abnormality detection, seizure monitoring, EMG gesture recognition, and silent speech—all deployed through a unified PyTorch → Brevitas QAT → GAP9 pipeline.
- **First real-time FM on GAP9:** CEReBrO achieves 146 ms latency and 8 mJ per inference; FEMBA is quantized down to 2-bit weights; TinyMyo runs at ~36 mW always-on; and LUNA brings topology-agnostic channel unification to the edge.
- **9× compression, minimal accuracy loss:** INT8 quantization across all models yields less than 2% accuracy drop on TUAB and CHB-MIT, with the NE16 neural accelerator pushing throughput beyond 10 MACs/cycle.
- **Edge-ready wearables:** BioGap integration projects up to 2 days of battery life, making continuous health monitoring with foundation-scale intelligence a practical reality.

## Photo gallery

{{< figure src="FM_Ultra_edge.jpeg" caption="Presenting the biosignal FM edge-deployment poster at AMLD Intelligence Summit 2026 in Lausanne." >}}

## Poster & resources

- [Download the poster PDF →](uploads/amld2026_fm_edge_poster.pdf)
- [BioFoundation GitHub repository →](https://github.com/pulp-bio/biofoundation)

## Related work

- [LUNA publication →](/publication/2025-neurips-luna/)
- [BioFoundation project →](/project/biofoundation/)
