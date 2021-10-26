---
abstract: In recent years, deep learning (DL) has contributed significantly to
  the improvement of motor-imagery brain-machine interfaces (MI-BMIs) based on
  electroencephalography (EEG). While achieving high classification accuracy, DL
  models have also grown in size, requiring a vast amount of memory and
  computational resources. This poses a major challenge to an embedded BMI
  solution that guarantees user privacy, reduced latency, and low power
  consumption by processing the data locally. In this paper, we propose
  EEG-TCNet, a novel temporal convolutional network (TCN) that achieves
  outstanding accuracy while requiring few trainable parameters. Its low memory
  footprint and low computational complexity for inference make it suitable for
  embedded classification on resource-limited devices at the edge. Experimental
  results on the BCI Competition IV-2a dataset show that EEG-TCNet achieves
  77.35% classification accuracy in 4-class MI. By finding the optimal network
  hyperparameters per subject, we further improve the accuracy to 83.84%.
  Finally, we demonstrate the versatility of EEG-TCNet on the Mother of All BCI
  Benchmarks (MOABB), a large scale test benchmark containing 12 different EEG
  datasets with MI experiments. The results indicate that EEG-TCNet successfully
  generalizes beyond one single dataset, outperforming the current
  state-of-the-art (SoA) on MOABB by a meta-effect of 0.25.
slides: ""
url_pdf: "https://arxiv.org/pdf/2006.00622.pdf"
publication_types:
  - "1"
tags:
- Deep Learning
authors:
  - admin
  - Michael Hersche
  - Xiaying Wang
  - Nobuaki Kobayashi
  - Lukas Cavigelli
  - Luca Benini
author_notes: []
publication: IEEE International Conference on Systems, Man, and Cybernetics
summary: In this paper, we propose EEG-TCNet, a novel temporal convolutional
  network (TCN) that achieves outstanding accuracy while requiring few trainable
  parameters.
url_dataset: ""
url_project: ""
publication_short: In *SMC*
url_source: ""
url_video: ""
title: "EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded
  Motor-Imagery Brain–Machine Interfaces"
doi: 10.1109/SMC42975.2020.9283028
featured: true
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured.png
date: 2020-10-11T00:00:00.000Z
url_slides: ""
publishDate: 2020-10-11T00:00:00.000Z
url_poster: ""
url_code: ""
---
Supplementary code for the paper can be found [here](https://github.com/iis-eth-zurich/eeg-tcnet).