---
title: Real-Time Drone Navigation with a Wearable EEG Headband
summary: We built a non-stigmatizing EEG headband with an ultra-low-power PULP processor that recognizes eye blinks, eyebrow movements, and alpha waves in real time and uses them to control a flying nano-drone — all computation happens on-board the wearable, without a laptop in the loop.
date: "2023-09-01T00:00:00Z"
tags:
- Brain-Computer Interfaces
- Edge AI
- Embedded Systems
- Biosignals
# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Wearable 8-channel EEG headband with integrated BioGAP module used for real-time drone control.
  focal_point: Smart

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/thorirmar
url_code: ""
url_pdf: "uploads/BCIAward2023.pdf"
url_slides: ""
url_video: "https://www.youtube.com/watch?v=1KPFJlJaXTI"

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

We developed a **wearable, non-stigmatizing EEG headband** that lets a user pilot a nano-drone in real time using heterogeneous EEG signals such as double eye blinks, left/right eye blinks, eyebrow raises, and alpha waves (eyes closed). All biosignal processing and classification run directly on a **BioGAP** module with a PULP **GAP9** ultra-low-power SoC, and the decoded commands are sent to the drone over Bluetooth Low Energy — no external laptop is required.

### System overview

The headband integrates:

- **8 dry EEG electrodes** (Datwyler) with active buffering, embedded inside a comfortable textile band.
- The **BioGAP** platform, combining:
  - a 10-core GAP9 RISC-V SoC for on-device signal processing and machine learning,
  - a BLE radio for wireless communication,
  - a medical-grade analog front-end for ExG acquisition.

The complete system continuously acquires EEG at 500 Hz, extracts spectral features on-device, classifies the user’s current mental/ocular state, and sends corresponding control packets to a **Crazyflie 2.1** nano-drone to steer it in an indoor environment.

### Signal processing and tiny-ML pipeline

We use a compact, interpretable feature pipeline tailored to ultra-low-power hardware:

- **Windowing:** 2-second sliding windows with overlapping updates.
- **Time–frequency features:**
  - 4-level **Discrete Wavelet Transform (DWT)** with Haar wavelets; energy is computed at each detail level per channel.
  - **FFT-based band energy** features (delta/theta, alpha, beta, high-frequency bands) per channel.
- **Feature vector:** 64+ engineered features across all 8 channels.
- **Classifier:** a hardware-aware **gradient-boosted tree model (XGBoost)** trained per user on six classes:
  - double eye blink,
  - left blink,
  - right blink,
  - eyebrow raise,
  - alpha (eyes closed),
  - neutral / baseline EEG.

The final model is compiled and deployed to GAP9, where inference takes **< 10 ms** per window, comfortably meeting real-time constraints.

### Drone control

The classifier outputs are mapped to high-level control commands for a Crazyflie nano-drone (e.g., take off / land, rotate left/right, move forward, stop). A BLE link between the headband and the drone enables:

- **Low-latency control** without any tethered devices.
- A fully portable setup where the user can walk around wearing only the headband.

The drone itself runs its standard firmware on an STM32F405 MCU, handling state estimation and motor control while accepting command packets from the headband.

### Performance and user experience

- **Accuracy:** >96% in subject-specific 6-class EEG classification.
- **Real-time operation:** end-to-end latency from EEG acquisition to drone actuation is well below 1 second, with signal classification on GAP9 in under 10 ms.
- **Comfort & aesthetics:** the textile headband hides wiring and electronics, avoiding the “medical cap” look and making the system more acceptable for everyday use.
- **Accessibility:** the interface is particularly promising for users with limited mobility, providing an intuitive, hands-free control modality.

### Demo video

A short demonstration of the system in action, showing real-time drone navigation controlled via EEG, is available here:

{{< youtube 1KPFJlJaXTI >}}