---
abstract: The development of a device for long-term and continuous monitoring of epilepsy is a very challenging objective, due to the high accuracy standards and nearly zero false alarms required by clinical practices. To comply with such requirements, most of the approaches in the literature rely on a high number of acquisition channels and exploit classifiers operating on pre-processed features, hand-crafted considering the available data, currently fairly limited. Thus, they lack comfort, portability, and adaptability to future use cases and datasets. A step forward is needed towards the implementation of unobtrusive, wearable systems, with a reduced number of channels, implementable on ultra-low-power computing platforms. Leveraging the promising ability of transformers in capturing long-term raw data dependencies in time series, we present in this work EEGformer, a compact transformer model for more adaptable seizure detection, that can be executed in real-time on tiny MicroController Units (MCUs) and operates on just the raw electroencephalography (EEG) signal acquired by the 4 temporal channels. Our proposed model is able to detect 73% of the examined seizure events (100% when considering 6 out of 8 patients), with an average onset detection latency of 15.2s. The False Positive/hour (FP/h) rate is equal to 0.8 FP/h, although 100% specificity is obtained in most tests, with 5/40 outliers that are mostly caused by EEG artifacts. We deployed our model on the Ambiq Apollo4 MCU platform, where inference run requires 405 ms and 1.79 mJ at 96 MHz operating frequency, demonstrating the feasibility of epilepsy detection on raw EEG traces for low-power wearable systems. Considering the CHB-MIT Scalp EEG dataset as a reference, we compare with a state-of-the-art classifier, acting on handcrafted features designed on the target dataset, reaching well-aligned accuracy results and reducing the onset detection latency by over 20%. Moreover, we compare with two adequately optimized Convolutional Neural Networks-based approaches, outperforming both alternatives on all the accuracy metrics.
slides: ""
url_pdf: "https://ieeexplore.ieee.org/abstract/document/9948637"
publication_types:
  - "1"
authors:
  - Paola Busia
  - Andrea Cossettini
  - admin
  - Simone Benatti
  - Alessio Burrello
  - Moritz Scherer
  - Matteo Antonio Scrugli
  - Paolo Meloni
  - Luca Benini
author_notes: []
publication: IEEE Biomedical Circuits and Systems Conference (BioCAS)
summary: EEGformer, a compact transformer model for seizure detection on raw EEG traces, demonstrating efficient execution on MCUs with competitive accuracy.
url_dataset: ""
url_project: ""
publication_short: In *BioCAS*
url_source: ""
url_video: ""
title: "EEGformer: Transformer-Based Epilepsy Detection on Raw EEG Traces for Low-Channel-Count Wearable Continuous Monitoring Devices"
doi: "10.1109/BioCAS54905.2022.9948637"
featured: false
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured.png
date: 2022-10-13T00:00:00.000Z
url_slides: ""
publishDate: 2022-10-13T00:00:00.000Z
url_poster: ""
url_code: ""
---