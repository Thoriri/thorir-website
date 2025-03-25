---
abstract: The long-term, continuous analysis of electroencephalography (EEG) signals on wearable devices to automatically detect seizures in epileptic patients is a high-potential application field for deep neural networks, and specifically for transformers, which are highly suited for end-to-end time series processing without handcrafted feature extraction. In this work, we propose a small-scale transformer detector, the EEGformer, compatible with unobtrusive acquisition setups that use only the temporal channels. EEGformer is the result of a hardware-oriented design exploration, aiming for efficient execution on tiny low-power micro-controller units (MCUs) and low latency and false alarm rate to increase patient and caregiver acceptance.Tests conducted on the CHB-MIT dataset show a 20% reduction of the onset detection latency with respect to the state-of-the-art model for temporal acquisition, with a competitive 73% seizure detection probability and 0.15 false-positive-per-hour (FP/h). Further investigations on a novel and challenging scalp EEG dataset result in the successful detection of 88% of the annotated seizure events, with 0.45 FP/h.We evaluate the deployment of the EEGformer on three commercial low-power computing platforms: the single-core Apollo4 MCU and the GAP8 and GAP9 parallel MCUs. The most efficient implementation (on GAP9) results in as low as 13.7 ms and 0.31 mJ per inference, demonstrating the feasibility of deploying the EEGformer on wearable seizure detection systems with reduced channel count and multi-day battery duration.
slides: ""
url_pdf: "https://ieeexplore.ieee.org/abstract/document/10412626"
publication_types:
  - "1"
authors:
  - Paola Busia
  - Andrea Cossettini
  - admin
  - Simone Benatti
  - Alessio Burrello
  - Victor J. B. Jung
  - Moritz Scherer
  - Matteo A. Scrugli
  - Adriano Bernini
  - Pauline Ducouret
  - Philippe Ryvlin
  - Paolo Meloni
  - Luca Benini
author_notes: []
publication: IEEE Transactions on Biomedical Circuits and Systems
summary: An extended version of EEGformer, demonstrating improved false alarm reduction and efficient deployment on various MCU platforms.
url_dataset: ""
url_project: ""
publication_short: In *IEEE TBioCAS*
url_source: ""
url_video: ""
title: "Reducing False Alarms in Wearable Seizure Detection With EEGformer: A Compact Transformer Model for MCUs"
doi: "10.1109/TBCAS.2024.3357509"
featured: false
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured.png
date: 2024-03-25T00:00:00.000Z
url_slides: ""
publishDate: 2024-03-25T00:00:00.000Z
url_poster: ""
url_code: ""
---