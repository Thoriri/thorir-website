---
title: "Learning World Models from Physiological Time Series"
summary: "Build and stress-test a self-supervised world model for biosignals, and work out where its anti-collapse regulariser stops respecting the fact that the data has an order."
type: student-project
status: "Open MSc Thesis"
date: 2026-07-31
categories: ["MSc Thesis"]
tags:
  - World Models
  - Self-Supervised Learning
  - JEPA
  - Time-Series
  - Biosignals
image:
  caption: "What a JEPA world model is, and where its training objective ignores time"
  focal_point: Center
  preview_only: false
featured: false
---

![Learning World Models from Physiological Time Series](/uploads/world_models_figure.jpg)

<div class="tm-collab">
  <span class="tm-collab-label">In collaboration with</span>
  <a href="https://www.timetracelabs.com" target="_blank" rel="noopener">
    <img src="/uploads/partner-logos/timetrace-logo.png" alt="TimeTraceLabs" loading="lazy">
  </a>
  <span class="tm-collab-note">Co-supervised by Blaise Delany</span>
</div>

## Overview

Much of the recent progress in AI comes from self-supervised learning. If you train a system to predict what comes next in a sequence, it has to learn a great deal about the sequence along the way. Applied to processes that unfold over time instead of to text, this recipe gives world models: systems that learn from experience how a process evolves, for example how a robot arm responds to a command, or how a patient's physiological state moves from one moment to the next.

Physiological data has no obvious equivalent of a word, and much of what a sensor records is noise a naive predictor would waste its capacity on. Joint-embedding predictive architectures (JEPAs) get around this by predicting a compressed internal summary of what is about to happen, rather than raw sensor values point by point. Training combines two terms: a prediction loss on future embeddings, and an anti-collapse regulariser that stops the encoder cheating by mapping every input to the same trivial vector. The regulariser most widely used today is SIGReg, which pushes the embedding distribution towards an isotropic Gaussian.

## The Open Problem

JEPAs have already been applied to biosignals, including ECG, EEG, fMRI and echocardiography, but almost all of that work relies on the older heuristic anti-collapse machinery: stop-gradients, EMA teachers, masked prediction. Whether the newer and more principled regularisers suit data that is autocorrelated and cyclostationary is still an open question, and it has two parts.

**Autocorrelation.** SIGReg's statistic is a normality test evaluated on a batch of embeddings. A reading taken half a second after another is nearly a copy of it, so a batch of overlapping windows contains far fewer effectively independent samples than its size suggests. A test calibrated for independent draws then reports a confidence it has not earned, and what that does to the resulting gradient has not been characterised.

**Progression.** Physiology is largely defined by its order. A heartbeat moves through a repeating cycle, sleep architecture progresses through distinct stages across a night, and a deteriorating patient moves along a trajectory over days. The regulariser asks for the same isotropic target at every point in that structure, and nothing in it distinguishes a window recorded at the QRS complex from one recorded during diastole.

This is a property of the regulariser and not of the architecture, since the prediction loss is explicitly temporal. So the question is specific: does an order-blind regulariser leave usable phase structure unexploited that the prediction term does not recover on its own, and what does that cost when the sensors misbehave?

## Directions

The project is deliberately broad rather than a fixed specification. What you pursue, and how far, is something we scope together in the first few weeks. A careful negative result counts: showing cleanly that one of these gaps does not matter in practice is a real and publishable finding.

- **Measure it.** Quantify how much of a recording's phase and progression survives inside representations trained the standard way, ideally with a probe cheap enough to run before committing to a full training run.
- **Design it.** Build and test a regulariser aware of physiological order, holding architecture and compute budget fixed against a standard baseline.
- **Interpret it.** Probe the latent space for physiologically meaningful axes: cardiac phase, respiratory cycle, sleep stage. Theory predicts alignment plus Gaussian regularisation should recover the underlying latent factors up to a linear map, and whether that holds on real recordings is directly testable.
- **Break it.** Desynchronise channels, drift the clocks, warp time, drop packets, then report where each method's advantage disappears. The failure boundary matters more than a benchmark row.

## What You Get

- A working JEPA world-model training pipeline, so the first weeks go into the problem rather than infrastructure
- Access to large-scale physiological datasets through our ongoing foundation-model work
- HPC and GPU cluster access sufficient for the full training and ablation programme
- Weekly supervision from both the sensor engineering and the AI side
- A realistic route to a workshop paper (NeurIPS or ICLR workshops on self-supervised learning, time series, or ML for health) alongside the thesis, plus an open-source release

## Prerequisites

- Comfortable in Python and PyTorch, and you have trained and debugged a neural network yourself
- Linear algebra and probability at MSc level
- A course in signal processing, time series or statistical testing helps but is not required
- No biomedical background needed

Suitable for MSc students in computer science, electrical engineering, data science, statistics, robotics or biomedical engineering. A reduced-scope semester project is also possible.

## Application

Please email `thoriri@iis.ee.ethz.ch` with the subject **"[MSc Thesis] Learning World Models from Physiological Time Series"**. Include a short paragraph on which part of the problem interests you and why, your CV and transcript, and a link to code you have written if you have any you are willing to show.

Happy to talk before you commit: a thirty-minute conversation costs nothing and can save us both a semester.

---

**Related Links**
- I-JEPA (Assran et al.): <https://arxiv.org/abs/2301.08243>
- LeJEPA, which introduces SIGReg (Balestriero & LeCun): <https://arxiv.org/abs/2511.08544>
- LeWorldModel, the pipeline to start from: <https://arxiv.org/abs/2603.19312>
- When Does LeJEPA Learn a World Model? (Klindt et al.): <https://arxiv.org/abs/2605.26379>
- Brain-JEPA, the same recipe on fMRI: <https://arxiv.org/abs/2409.19407>
- Project page on the IIS wiki: <https://iis-projects.ee.ethz.ch/index.php?title=Learning_World_Models_from_Physiological_Time_Series>
