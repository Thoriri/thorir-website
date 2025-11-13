---
widget: blank
headless: true
weight: 25

title: "What I'm Currently Working On"
subtitle: ""

design:
  columns: '1'
  spacing:
    padding: ['40px', '0', '40px', '0']
---

<div style="margin-bottom: 20px;">
As a researcher at the intersection of TinyML and biomedical AI, I'm currently exploring:
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin-top: 30px;">

<div class="research-card">

### 🧠 Foundation Models for Biosignals

I'm developing large-scale pre-trained models that can understand diverse biomedical signals with minimal fine-tuning. Our **LUNA** model (NeurIPS 2025) achieves topology-agnostic EEG analysis with 300× fewer FLOPs and 10× less memory than traditional approaches, enabling more robust and generalizable health monitoring systems.

**Recent work:** [LUNA at NeurIPS 2025](https://arxiv.org/abs/2510.22257) | [Code on GitHub](https://github.com/pulp-bio/biofoundation)

</div>

<div class="research-card">

### 🔄 Tiny Recursion Models

I'm investigating how deep recursion and supervision techniques can be applied to time-series biosignals to improve model efficiency and accuracy. This approach enables more sophisticated temporal modeling while maintaining the ultra-low computational budgets required for edge deployment.

**Focus areas:** Deep supervision, recurrent architectures, temporal feature learning

</div>

<div class="research-card">

### 🚀 Edge AI Deployment

I'm developing hardware-aware methods to deploy foundation models and advanced ML systems on resource-constrained wearable devices. This involves co-designing algorithms and implementations to achieve microwatt-level power consumption while maintaining clinical-grade performance for applications like seizure detection and physiological monitoring.

**Technologies:** GAP9, RISC-V processors, TinyML optimization

</div>

</div>
