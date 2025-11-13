Title:
Quantized Tiny Recursive Models for Edge Deployment

Short Title (if needed for listings):
Quantized TRMs on Edge Devices

Summary (1–2 sentences, Markdown allowed):
Tiny Recursive Models (TRMs) are naturally small and recurrent, making them promising candidates for ultra-efficient deployment on microcontrollers and edge devices. In this project, you will quantize TRMs to low precision (e.g., INT8/INT4), study their stability under recursion and deep supervision, and deploy them on a real embedded platform (e.g., GAP9 or Cortex-M) to explore accuracy–energy trade-offs and adaptive compute via learned halting.

Project Type:
MSc thesis

Duration:
6 months

Status:
Open

Primary Supervisor(s):
- Thorir Már Ingólfsson (Integrated Systems Laboratory IIS, ETH Zurich)

Co-supervisor(s) / Lab mentor(s) (optional):
- tbd (e.g., a PULP / embedded AI PhD student in IIS)

Key Topics / Keywords:
tiny recursive models, quantization, low-precision inference, edge AI, microcontrollers, GAP9, PULP, deep supervision, adaptive compute, halting mechanisms, efficient deep learning, time-series classification

Expected Outcomes:
- A reference PyTorch implementation of a Tiny Recursive Model suitable for quantization (based on an existing TRM or the companion “TRM for time-series” thesis).
- Post-training and quantization-aware training (QAT) pipelines (e.g., INT8, possibly INT4/mixed precision) for TRMs, with detailed accuracy–precision curves.
- Systematic study of how quantization affects recursive dynamics: stability across steps, error accumulation, and the role of deep supervision.
- A deployed implementation of a quantized TRM on a chosen edge platform (e.g., GAP9 or Cortex-M), including measurement of latency, memory footprint, and energy per inference.
- Evaluation of dynamic halting on-device, showing compute/energy savings for “easy” examples versus fixed-compute baselines.
- Optional: short technical report or workshop paper summarizing the quantization and deployment findings.

Prerequisites:
- Strong programming skills in Python and C/C++.
- Basic background in deep learning and neural network training (PyTorch or equivalent).
- Interest in embedded systems / edge AI, and willingness to learn toolchains for real hardware.
- Prior exposure to quantization or model compression is a plus but not required.

Tools & Skills:
- Python, PyTorch (and optionally PyTorch quantization / Brevitas / other QAT frameworks).
- C/C++ for embedded deployment (e.g., GAP SDK / PMSIS, or an ARM microcontroller toolchain).
- Basic Linux, Git, and scripting for experiment automation.
- Tools for power/latency measurement on the selected hardware platform (to be provided in the lab).

What you will learn:
- How Tiny Recursive Models operate and how their recursive structure interacts with low-precision arithmetic.
- Practical techniques for post-training quantization and quantization-aware training of sequence / recursive models.
- How to map a compact neural network efficiently onto a real microcontroller or PULP-style SoC, including memory and tiling considerations.
- How to measure and interpret energy, latency, and accuracy trade-offs for edge deployment.
- Experience bridging ML algorithms and embedded systems, culminating in a full-stack MSc thesis.

How to apply / Call to action:
Please send a short motivation paragraph, your CV, and grade transcripts to `thoriri@iis.ee.ethz.ch` with the subject line **"[MSc Thesis] Quantized Tiny Recursive Models for Edge Deployment"**. Mention any prior experience with PyTorch, quantization, or embedded / PULP platforms (if any).

Optional extras:
- Links (paper/code/demo) with URLs
  - Tiny Recursive Models (TRM) paper by Jolicoeur-Martineau et al.: <https://arxiv.org/abs/2510.04871>
  - PULP platform overview: <https://pulp-platform.org/>
- Preferred start date:
  - Flexible in 2025 (to be agreed individually).
- Related publications or projects:
  - IIS work on efficient biosignal foundation models and their deployment on MCUs (e.g., GAP9).
- Image filename:
  - trm_quantized_edge.png
