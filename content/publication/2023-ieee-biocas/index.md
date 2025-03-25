---
abstract: In ski jumping, low repetition rates of jumps limit the effectiveness of training. Thus, increasing learning rate within every single jump is key to success. A critical element of athlete training is motor learning, which has been shown to be accelerated by feedback methods. In particular, a fine-grained control of the center of gravity in the in-run is essential. This is because the actual takeoff occurs within a blink of an eye (∼ 300 ms), thus any unbalanced body posture during the in-run will affect flight.This paper presents a smart, compact, and energy-efficient wireless sensor system for real-time performance analysis and biofeedback during ski jumping. The system operates by gauging foot pressures at three distinct points on the insoles of the ski boot at 100 Hz. Foot pressure data can either be directly sent to coaches to improve their feedback, or fed into a Machine Learning (ML) model to give athletes instantaneous in-action feedback using a vibration motor in the ski boot. In the biofeedback scenario, foot pressures act as input variables for an optimized XGBoost model. We achieve a high predictive accuracy of 92.7 % for center of mass predictions (dorsal shift, neutral stand, ventral shift). Subsequently, we parallelized and fine-tuned our XGBoost model for a RISC-V based low power parallel processor (GAP9), based on the Parallel Ultra-Low Power (PULP) architecture. We demonstrate real-time detection and feedback (0.0109 ms/inference) using our on-chip deployment. The proposed smart system is unobtrusive with a slim form factor (13 mm baseboard, 3.2 mm antenna) and a lightweight build (26 g). Power consumption analysis reveals that the system's energy-efficient design enables sustained operation over multiple days (up to 300 hours) without requiring recharge.
slides: ""
url_pdf: "https://ieeexplore.ieee.org/abstract/document/10389124"
publication_types:
  - "1"
authors:
  - Lukas Schulthess
  - admin
  - Marc Nölke
  - Michele Magno
  - Luca Benini
  - Christoph Leitner
author_notes: []
publication: IEEE Biomedical Circuits and Systems Conference (BioCAS)
summary: A smart sensor system for real-time performance analysis and biofeedback in ski jumping, featuring energy-efficient design and ML-based predictions.
url_dataset: ""
url_project: ""
publication_short: In *BioCAS*
url_source: ""
url_video: ""
title: "Skilog: A Smart Sensor System for Performance Analysis and Biofeedback in Ski Jumping"
doi: "10.1109/BioCAS58349.2023.10389124"
featured: false
tags: []
projects: []
image:
  caption: ""
  focal_point: Smart
  preview_only: false
  filename: featured.png
date: 2023-10-19T00:00:00.000Z
url_slides: ""
publishDate: 2023-10-19T00:00:00.000Z
url_poster: ""
url_code: ""
---