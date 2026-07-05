---
active: false
widget: blank
headless: true
weight: 35

title: "Featured Work"
subtitle: ""

design:
  columns: '1'
  spacing:
    padding: ['40px', '0', '40px', '0']
---

<div class="featured-work-section" style="margin-bottom: 30px;">

## 🏆 LUNA: Foundation Model for EEG Analysis

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 20px;">

<div>

**NeurIPS 2025** | [📄 Read Paper](https://arxiv.org/abs/2510.22257)

LUNA is an efficient, topology-agnostic foundation model for EEG signal analysis that reconciles disparate electrode configurations while achieving unprecedented computational efficiency.

**Key Achievements:**
- 🎯 **300× reduction in FLOPs** compared to standard transformers
- 💾 **10× less GPU memory** usage
- 🏅 **State-of-the-art performance**: 0.921 AUROC on TUAR benchmark
- 📊 **Pretrained on 21,000+ hours** of diverse EEG data
- 🌍 **Topology-agnostic**: Works across different electrode layouts

The model uses learned queries and cross-attention mechanisms to compress multi-channel EEG into a unified latent representation, enabling practical deployment of foundation models for biosignals.

</div>

<div>

**🔗 Resources**

<div class="resource-box">
<strong>GitHub Repository</strong><br>
<a href="https://github.com/pulp-bio/biofoundation" style="font-size: 14px;">github.com/pulp-bio/biofoundation</a><br>
⭐ 121 stars • 🍴 15 forks • Apache 2.0 License
</div>

<div class="resource-box">
<strong>Key Features</strong>
<ul>
<li>PyTorch Lightning implementation</li>
<li>Hydra configuration system</li>
<li>Pre-trained model weights available</li>
<li>FEMBA & LUNA architectures</li>
<li>Distributed training support</li>
</ul>
</div>

<div class="resource-box">
<strong>Applications</strong>
<ul>
<li>Abnormality detection</li>
<li>Artifact rejection</li>
<li>Slowing classification</li>
<li>Emotion recognition</li>
</ul>
</div>

</div>

</div>

<div style="margin-top: 20px; text-align: center;">
<a href="https://arxiv.org/abs/2510.22257" class="btn btn-primary" style="margin-right: 10px;">📄 Read the Paper</a>
<a href="https://github.com/pulp-bio/biofoundation" class="btn" style="background: var(--nordic-primary); color: white; border-color: var(--nordic-primary);">💻 View Code</a>
</div>

</div>
