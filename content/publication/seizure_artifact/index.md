---
abstract: Electroencephalography (EEG) is widely used to monitor epileptic seizures, and standard clinical practice consists of monitoring patients in dedicated epilepsy monitoring units via video surveillance and cumbersome EEG caps. Such a setting is not compatible with long-term tracking under typical living conditions, thereby motivating the development of unobtrusive wearable solutions. However, wearable EEG devices present the challenges of fewer channels, restricted computational capabilities, and lower signal-to-noise ratio. Moreover, artifacts presenting morphological similarities to seizures act as major noise sources and can be misinterpreted as seizures. This paper presents a combined seizure and artifacts detection framework targeting wearable EEG devices based on Gradient Boosted Trees. The seizure detector achieves nearly zero false alarms with average sensitivity values of 65.27% for 182 seizures from the CHB-MIT dataset and 57.26% for 25 seizures from the private dataset with no preliminary artifact detection or removal. The artifact detector achieves a state-of-the-art accuracy of 93.95% (on the TUH-EEG Artifact Corpus dataset). Integrating artifact and seizure detection significantly reduces false alarms—up to 96% compared to standalone seizure detection.  Optimized for a Parallel Ultra-Low Power platform, these algorithms enable extended monitoring with a battery lifespan reaching 300 hours. These findings highlight the benefits of integrating artifact detection in wearable epilepsy monitoring devices to limit the number of false positives.
slides: ""
url_pdf: "https://www.nature.com/articles/s41598-024-52551-0"
publication_types:
  - "1"
authors:
  - admin
  - Simone Benatti
  - Xiaying Wang
  - Adriano Bernini
  - Pauline Ducouret
  - Philippe Ryvlin
  - Sandor Beniczky
  - Luca Benini
  - Andrea Cossettini
author_notes: []
publication: Nature Scientific Reports
summary: The paper presents a combined seizure and artifact detection framework based on Gradient Boosted Trees. The framework achieves high accuracy in detecting seizures and artifacts, reducing false alarms. The algorithms are optimized for a Parallel Ultra-Low Power platform, enabling extended monitoring with a long battery lifespan. The paper highlights the benefits of integrating artifact detection in wearable epilepsy monitoring devices.
url_dataset: ""
url_project: ""
publication_short: In *Nature*
url_source: ""
url_video: ""
title: "Minimizing artifact-induced false-alarms for seizure detection in wearable EEG devices with gradient-boosted tree classifiers"
doi: 
featured: true
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured-1.png
date: 2024-02-05T00:00:00.000Z
url_slides: ""
publishDate: 2024-02-05T00:00:00.000Z
url_poster: ""
url_code: ""
---
