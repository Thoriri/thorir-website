---
abstract: Epilepsy is a prevalent neurological disorder that affects millions of individuals globally, and continuous monitoring coupled with automated seizure detection appears as a necessity for effective patient treatment. To enable long-term care in daily-life conditions, comfortable and smart wearable devices with long battery life are required, which in turn set the demand for resource-constrained and energy-efficient computing solutions. In this context, the development of machine learning algorithms for seizure detection faces the challenge of heavily imbalanced datasets. This paper introduces EpiDeNet, a new lightweight seizure detection network, and Sensitivity-Specificity Weighted Cross-Entropy (SSWCE), a new loss function that incorporates sensitivity and specificity, to address the challenge of heavily unbalanced datasets. The proposed EpiDeNet-SSWCE approach demonstrates the successful detection of 91.16% and 92.00% seizure events on two different datasets (CHB-MIT and PEDESITE, respectively), with only four EEG channels. A three-window majority voting-based smoothing scheme combined with the SSWCE loss achieves 3x reduction of false positives to 1.18 FP/h. EpiDeNet is well suited for implementation on low-power embedded platforms, and we evaluate its performance on two ARM Cortex-based platforms (M4F/M7) and two parallel ultra-low power (PULP) systems (GAP8, GAP9). The most efficient implementation (GAP9) achieves an energy efficiency of 40 GMAC/s/W, with an energy consumption per inference of only 0.051 mJ at high performance (726.46 MMAC/s), outperforming the best ARM Cortex-based solutions by approximately 160x in energy efficiency. The EpiDeNet-SSWCE method demonstrates effective and accurate seizure detection performance on heavily imbalanced datasets, while being suited for implementation on energy-constrained platforms.
slides: ""
url_pdf: "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/649848/EpiDeNet____BioCAS_Paper____Arxiv_Version%281%29.pdf?sequence=1&isAllowed=y"
publication_types:
  - "1"
authors:
  - admin
  - Upasana Chakraborty
  - Xiaying Wang
  - Sandor Beniczky
  - Pauline Ducouret
  - Simone Benatti
  - Philippe Ryvlin
  - Andrea Cossettini
  - Luca Benini
author_notes: []
publication: IEEE International Conference on Biomedical Circuits and Systems
summary: In this paper, we introduce EpiDeNet, a lightweight seizure detection network, and a novel loss function (SSWCE) to address imbalanced datasets, achieving high accuracy, reduced false positives, and energy-efficient performance on low-power embedded platforms.
url_dataset: ""
url_project: ""
publication_short: In *BIOCAS*
url_source: ""
url_video: ""
title: "EpiDeNet: An Energy-Efficient Approach to Seizure Detection for Embedded Systems"
doi: 
featured: true
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured.png
date: 2023-10-06T00:00:00.000Z
url_slides: ""
publishDate: 2023-10-06T00:00:00.000Z
url_poster: ""
url_code: ""
---
