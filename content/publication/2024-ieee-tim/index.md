---
abstract: Measurement of muscle activity via wearable instrumentation is of great interest for medical and sport science applications: examples include the control of prostheses, robotics, or therapy and training. This article focuses on the measurement of pennation angles in lateral gastrocnemius (LG) muscles directly from raw ultrasound (US) data. We rely on a reduced number of channels (32) to demonstrate accurate and real-time-compatible predictions of pennation angles for low-power wearable US devices. B-mode images are reconstructed by means of a delay-and-sum beamformer, and an automated computer vision tool (AEPUS) is employed to perform annotations. The labels are then used as ground truth for training an XGBoost regressor based on statistical features extracted from raw US data. Feature importance analyses and deployment-oriented XGBoost model design allow reducing the memory footprint down to only 11 kB. Experimental verification demonstrates a root-mean-square error (RMSE) of 1.6°, comparable to manual annotations by experts. To evaluate the performance on a low-power embedded processor, the complete algorithm is implemented on a recently released RISC-V-based parallel low-power processor (GAP9, from Greenwaves Technologies). Experimental measurements show an inference time and energy consumption as low as 1.31 ms and 43.32 μJ , respectively, with an average power envelope of 33.14 mW and a peak power of only 44 mW. The achieved accuracy, low memory footprint, and low power demonstrate the feasibility of pennation angle measurements directly on the probe for resource-constrained smart wearable US devices with a small number of channels.
slides: ""
url_pdf: "https://ieeexplore.ieee.org/abstract/document/10329945"
publication_types:
  - "1"
authors:
  - Sergei Vostrikov
  - admin
  - Soley Hafthorsdottir
  - Christoph Leitner
  - Michele Magno
  - Luca Benini
  - Andrea Cossettini
author_notes: []
publication: IEEE Transactions on Instrumentation and Measurement
summary: A framework for estimating muscle pennation angles from raw ultrasound data, demonstrating efficient implementation on low-power processors.
url_dataset: ""
url_project: ""
publication_short: In *IEEE TIM*
url_source: ""
url_video: ""
title: "A Muscle Pennation Angle Estimation Framework From Raw Ultrasound Data for Wearable Biomedical Instrumentation"
doi: "10.1109/TIM.2023.3335535"
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