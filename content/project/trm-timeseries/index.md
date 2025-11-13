---
title: "Tiny Recursive Models for Time-Series"
summary: "Adapt Tiny Recursive Models (TRMs) to non-visual time-series domains and analyse recursion, deep supervision, and adaptive compute."
type: project
status: "Open MSc Thesis"
date: 2025-01-01
categories: ["MSc Thesis"]
tags:
  - Tiny Recursive Models
  - Time-Series
  - Deep Supervision
  - Adaptive Compute
  - Edge AI
image:
  caption: "Tiny Recursive Models for biosignal time-series"
  focal_point: Center
  preview_only: false
featured: false
---

![Tiny Recursive Models for Time-Series](/uploads/trm_time_series.png)

## Overview

Tiny Recursive Models (TRMs) are small, deeply supervised recurrent-like architectures that refine their predictions step-by-step. This project adapts TRMs to non-visual domains such as physiological time-series (UCR archive, EEG, human activity recognition) and studies how recursion depth, deep supervision, and learned halting impact accuracy, robustness, and computation.

## Expected Outcomes

- PyTorch implementation of a TRM tailored to at least one non-visual dataset (e.g., UCR time-series, EEG, HAR)
- Benchmarks against capacity-matched baselines (tiny CNNs, Transformers, SSMs) under equal parameter / FLOP budgets
- Ablation study on recursion depth, deep supervision, and halting (learned vs. fixed)
- Analysis of per-sample adaptive compute and its relationship to task difficulty
- (Optional) Extension to forecasting or symbolic reasoning, and/or workshop / conference submission draft

## Prerequisites

- Strong background in machine learning and deep learning (completed introductory DL course)
- Solid Python programming skills and experience with PyTorch (or similar frameworks)
- Familiarity with time-series data *or* willingness to ramp up quickly
- Comfortable using Linux, Git, and running GPU experiments

## Tools & Skills

- Python, PyTorch (PyTorch Lightning welcome)
- Experiment tracking (e.g., Weights & Biases), NumPy, pandas, matplotlib
- Time-series preprocessing: resampling, normalisation, segmentation
- (Optional) Exposure to EEG / biosignal datasets

## What You Will Learn

- Inner workings of TRMs (deep supervision at each recursion, learned halting)
- How to adapt a research architecture from vision to time-series and biosignals
- Designing fair baselines and ablation studies for capacity-constrained models
- Analysing adaptive compute and visualising prediction evolution across recursion steps
- Writing research-quality code and producing an MSc thesis (potentially a joint publication)

## Application

Please email `thoriri@iis.ee.ethz.ch` with the subject **"[MSc Thesis] Tiny Recursive Models for Time-Series"**. Include a short motivation paragraph, CV, transcripts, and mention any prior experience with time-series, PyTorch, or efficient deep learning.

---

**Related Links**
- TRM paper (Jolicoeur-Martineau et al.): <https://arxiv.org/abs/2510.04871>
- IIS foundation model work (LUNA, FEMBA, CEReBrO)
