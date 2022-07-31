---
abstract: In the context of epilepsy monitoring, EEG artifacts are often mistaken for seizures due to their morphological similarity in both amplitude and frequency, making seizure detection systems susceptible to higher false alarm rates. In this work we present the implementation of an artifact detection algorithm based on a minimal number of EEG channels on a parallel ultra-low-power (PULP) embedded platform. The analyses are based on the TUH EEG Artifact Corpus dataset and focus on the temporal electrodes. First, we extract optimal feature models in the frequency domain using an automated machine learning framework, achieving a 93.95% accuracy, with a 0.838 F1 score for a 4 temporal EEG channel setup. The achieved accuracy levels surpass state-of-the-art by nearly 20%. Then, these algorithms are parallelized and optimized for a PULP platform, achieving a 5.21 times improvement of energy-efficient compared to state-of-the-art low-power implementations of artifact detection frameworks. Combining this model with a low-power seizure detection algorithm would allow for 300h of continuous monitoring on a 300 mAh battery in a wearable form factor and power budget. These results pave the way for implementing affordable, wearable, long-term epilepsy monitoring solutions with low false-positive rates and high sensitivity, meeting both patients' and caregivers' requirements.
slides: ""
url_pdf: "https://arxiv.org/pdf/2204.09577.pdf"
publication_types:
  - "1"
authors:
  - admin
  - Andrea Cossettini
  - Simone Benatti
  - Luca Benini
author_notes: []
publication: International Engineering in Medicine and Biology Conference
summary: In this paper, we present implementations of energy-efficient artifact detection algorithms on a parallel ultra-low power platform.
url_dataset: ""
url_project: ""
publication_short: In *EMBC*
url_source: ""
url_video: ""
title: "Energy-Efficient Tree-Based EEG Artifact Detection"
doi: 
featured: true
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured.png
date: 2022-06-06T00:00:00.000Z
url_slides: ""
publishDate: 2022-06-06T00:00:00.000Z
url_poster: ""
url_code: ""
---
