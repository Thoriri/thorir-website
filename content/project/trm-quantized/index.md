---
title: "Quantized Tiny Recursive Models for Edge Deployment"
summary: "Quantize TRMs to low precision and deploy them on GAP9 or Cortex-M, exploring accuracy–energy trade-offs and adaptive compute."
type: student-project
status: "Completed"
date: 2025-01-01
categories: ["MSc Thesis"]
tags:
  - Tiny Recursive Models
  - Quantization
  - Edge AI
  - Microcontrollers
  - GAP9
image:
  caption: "Quantized TRMs on ultra-low-power hardware"
  focal_point: Center
  preview_only: false
featured: false
---

![Quantized TRMs for Edge Deployment](/uploads/trm_quantized.jpg)

> **Completed.** This project ran as a Master thesis with **Wajeeha Tahir** and finished successfully. It produced the preprint [*Quantizing Recursive Reasoning Models*](https://arxiv.org/abs/2607.16237) (arXiv:2607.16237). The central finding: because a recursive model reuses one weight-tied block across many refinement steps, quantization error compounds at every step. Per-tensor 4-bit activation quantization accumulates a systematic bias that collapses exact-solution accuracy on Sudoku from 84.1% to 0.0%. The culprit is activation-scaling *granularity* rather than bit-width or number format, and blockwise scaling (MXInt4) fully restores accuracy.

## Overview

Tiny Recursive Models (TRMs) are naturally compact and recur over their latent state, making them attractive for ultra-efficient deployment on microcontrollers. This thesis develops low-precision (INT8/INT4) TRMs, analyses their stability under recursion and deep supervision, and deploys them on real hardware (GAP9 or Cortex-M) to study latency, memory, and energy trade-offs together with adaptive halting.

## Expected Outcomes

- Reference PyTorch TRM implementation ready for quantisation (building on the time-series thesis or existing TRM code)
- Post-training and quantisation-aware training (QAT) pipelines (INT8, INT4 / mixed precision) with detailed accuracy–precision curves
- Investigation of how quantisation affects recursive dynamics and deep supervision signals
- Deployment of a quantised TRM on GAP9 or Cortex-M, with measured latency, memory footprint, and energy per inference
- Evaluation of learned halting on-device versus fixed-compute baselines
- (Optional) Technical report or workshop paper summarising quantisation + deployment findings

## Prerequisites

- Strong programming skills in Python and C/C++
- Solid ML/DL foundations (PyTorch or equivalent)
- Interest in embedded systems / edge AI and willingness to work with hardware toolchains
- Prior experience with quantisation or model compression is a plus but not required

## Tools & Skills

- Python, PyTorch, quantisation frameworks (PyTorch built-ins, Brevitas, etc.)
- C/C++ for embedded deployment (GAP SDK / PMSIS, or ARM microcontroller toolchain)
- Linux, Git, scripting for experiment automation
- Access to lab tools for power/latency measurements (provided)

## What You Will Learn

- How TRMs behave under low-precision arithmetic and recursive refinement
- Practical post-training and QAT techniques for sequence / recursive models
- Mapping compact neural networks onto MCUs or PULP SoCs while respecting memory and timing constraints
- Measuring and interpreting energy/latency/accuracy trade-offs for edge deployment
- Integrating ML algorithms with embedded software stacks to deliver a full-stack thesis

---

**Related Links**
- TRM paper (Jolicoeur-Martineau et al.): <https://arxiv.org/abs/2510.04871>
- PULP Platform overview: <https://pulp-platform.org/>
- IIS foundation model work (LUNA, FEMBA, CEReBrO)
