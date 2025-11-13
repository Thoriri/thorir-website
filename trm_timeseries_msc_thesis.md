Title:
Tiny Recursive Models for Time-Series and Non-Visual Tasks

Short Title (if needed for listings):
Tiny Recursive Models for Time-Series

Summary (1–2 sentences, Markdown allowed):
Tiny Recursive Models (TRMs) are tiny, deeply supervised recurrent-like architectures that outperform much larger models on reasoning tasks by recursively refining their predictions. In this project, you will adapt TRMs to non-visual domains such as time-series classification (e.g., physiological signals, UCR datasets) and systematically study how recursion and deep supervision impact accuracy, generalization, and computation.

Project Type:
MSc thesis

Duration:
6 months

Status:
Open

Primary Supervisor(s):
- Thorir Már Ingólfsson (Integrated Systems Laboratory IIS, ETH Zurich)

Co-supervisor(s) / Lab mentor(s) (optional):
- tbd (depending on final task focus, e.g., EEG vs generic time-series)

Key Topics / Keywords:
tiny recursive models, deep supervision, deep recursion, time-series classification, sequence modelling, efficient deep learning, halting mechanisms, reasoning, EEG, edge AI

Expected Outcomes:
- A PyTorch implementation of Tiny Recursive Models adapted to at least one non-visual domain (e.g., UCR time-series, EEG, human activity recognition).
- A thorough benchmark comparing TRMs to capacity-matched baselines (e.g., tiny CNNs, Transformers, SSMs) under the same parameter and FLOP budget.
- Ablation study on recursion depth, deep supervision, and halting (learned vs fixed) and their effect on accuracy, robustness, and compute.
- Analysis of per-sample adaptive compute (how many recursion steps different examples require) and its relationship to task difficulty.
- Optional: extension to a second domain (forecasting or simple symbolic/tabular reasoning) and/or a workshop / conference publication draft.

Prerequisites:
- Solid background in machine learning and deep learning (e.g., completed an introductory DL course).
- Strong Python programming skills and experience with PyTorch or a similar framework.
- Familiarity with time-series data (e.g., signal processing, sequence models) *or* willingness to learn quickly.
- Comfortable working in Linux environments, using Git, and running experiments on GPUs.

Tools & Skills:
- Python, PyTorch (optionally PyTorch Lightning).
- Experiment tracking tools (e.g., Weights & Biases) and standard ML tooling (NumPy, pandas, matplotlib).
- Basic knowledge of time-series datasets and preprocessing (resampling, normalization, segmentation).
- Optional: familiarity with EEG or biosignal datasets.

What you will learn:
- How Tiny Recursive Models work in detail (recursive latent refinement, deep supervision at every step, learned halting).
- Practical skills for adapting a research architecture from one domain (visual reasoning) to others (time-series, biosignals).
- How to design fair baselines and ablation studies for capacity-constrained models.
- How to analyze adaptive compute, interpret model dynamics across recursion steps, and visualize evolving predictions.
- Experience in writing research-quality code and a master’s thesis, with potential for a joint publication.

How to apply / Call to action:
Please send a short motivation paragraph, your CV, and grade transcripts to `thoriri@iis.ee.ethz.ch` with the subject line **"[MSc Thesis] Tiny Recursive Models for Time-Series"**. Mention any prior experience with time-series, PyTorch, or efficient deep learning.

Optional extras:
- Links (paper/code/demo) with URLs
  - Tiny Recursive Models (TRM) paper by Jolicoeur-Martineau et al.: <https://arxiv.org/abs/2510.04871>
- Preferred start date:
  - Flexible in 2025 (discussable).
- Related publications or projects:
  - Efficient foundation models for biosignals at IIS (e.g., LUNA, FEMBA, CEReBrO).
- Image filename:
  - trm_timeseries.png
