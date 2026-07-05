---
title: "Forgis Research Night – Learning to Dream in EEG"

event: "Forgis Research Night"
event_url: ""
tags:
  - Forgis
  - Foundation Models
  - EEG
  - Self-Supervised Learning
  - LeJEPA
  - World Models
location: "DARE Campus (The JED), Zürich-Schlieren, Switzerland"
address:
  street: "The JED"
  city: Schlieren
  region: Canton of Zürich
  postcode: '8952'
  country: Switzerland

summary: Lightning talk on LuMamba — using LeJEPA "world-model" self-supervision to pretrain a tiny, montage-agnostic foundation model for biosignals.
abstract: |
  Invited lightning talk at Forgis Research Night (DARE Campus, Schlieren). I introduced **LuMamba**: a 4.6M-parameter EEG foundation model that pairs LUNA's channel-unification module — learned queries that read any electrode montage into a fixed latent space — with a linear-time bidirectional Mamba backbone. The core idea is to "learn to dream": instead of reconstructing the raw, noisy signal, we borrow the representation-learning half of LeCun's world-model recipe and predict in latent space with **LeJEPA**, kept collapse-free by the SIGReg (Sketched Isotropic Gaussian) regularizer. Mixing a little masked reconstruction with LeJEPA keeps clusters *and* generalizes, reaching state-of-the-art Alzheimer's detection (0.97 AUPR) — strongest on montages never seen during pretraining.

date: "2026-06-24T16:00:00+02:00"
all_day: false

publishDate: "2026-06-25T00:00:00Z"

authors: []

featured: true

image:
  caption: "Learning to Dream in EEG — Forgis Research Night, DARE Campus"
  focal_point: Smart
  filename: featured.jpeg
  preview_only: false

links:
  - icon: external-link-alt
    icon_pack: fas
    name: LUNA paper (arXiv)
    url: "https://arxiv.org/abs/2510.22257"
url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: ""
url_slides: "uploads/forgis2026_slides.pdf"
url_video: ""

slides: ""

projects: []
---

A short lightning talk on how we pretrain foundation models for biosignals — and why it pays to teach the model to *dream* rather than to copy.

## What I covered

- **The problem:** EEG is a tiny (~20 µV) signal buried in noise (SNR < 1), and every dataset uses a different electrode montage.
- **LuMamba:** LUNA channel-unification (learned queries → fixed latent, reads any montage) + a linear-time bidirectional Mamba backbone — 4.6M parameters.
- **Learning to dream:** rather than reconstructing the raw signal, we predict in latent space with **LeJEPA**, kept collapse-free by SIGReg. A mixed reconstruction + LeJEPA objective keeps cluster structure *and* generalizes — state-of-the-art Alzheimer's detection (0.97 AUPR), strongest on unseen montages.

## Slides & resources

- [Download the slides (PDF) →](uploads/forgis2026_slides.pdf)
- [BioFoundation codebase →](https://github.com/pulp-bio/biofoundation)
- [LUNA paper (arXiv) →](https://arxiv.org/abs/2510.22257)
- [Deep-dive blog post: LeJEPA EEG pretraining →](/post/lejepa_eeg_pretraining/)

## Photos

{{< figure src="Forgis_screen.jpeg" caption="Kicking off — *Learning to Dream in EEG* at DARE Campus." >}}

{{< figure src="Forgis_room.jpeg" caption="A full room for Forgis Research Night." >}}
