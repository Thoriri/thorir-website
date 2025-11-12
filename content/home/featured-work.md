---
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

<div style="border: 2px solid #3b82f6; border-radius: 12px; padding: 40px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); margin-bottom: 30px;">

## 🏆 LUNA: Foundation Model for EEG Analysis

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 20px;">

<div>

**Accepted at NeurIPS 2024** | [📄 Read Paper](https://arxiv.org/abs/2510.22257)

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

<div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
<strong>GitHub Repository</strong><br>
<a href="https://github.com/pulp-bio/biofoundation" style="font-size: 14px;">github.com/pulp-bio/biofoundation</a><br>
⭐ 27 stars • 🍴 2 forks • Apache 2.0 License
</div>

<div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
<strong>Key Features</strong>
<ul style="margin: 10px 0; padding-left: 20px;">
<li>PyTorch Lightning implementation</li>
<li>Hydra configuration system</li>
<li>Pre-trained model weights available</li>
<li>FEMBA & LUNA architectures</li>
<li>Distributed training support</li>
</ul>
</div>

<div style="background: white; padding: 20px; border-radius: 8px;">
<strong>Applications</strong>
<ul style="margin: 10px 0; padding-left: 20px;">
<li>Abnormality detection</li>
<li>Artifact rejection</li>
<li>Slowing classification</li>
<li>Emotion recognition</li>
</ul>
</div>

</div>

</div>

<div style="margin-top: 20px; text-align: center;">
<a href="https://arxiv.org/abs/2510.22257" style="display: inline-block; background: #3b82f6; color: white; padding: 12px 30px; border-radius: 6px; text-decoration: none; margin-right: 10px;">📄 Read the Paper</a>
<a href="https://github.com/pulp-bio/biofoundation" style="display: inline-block; background: #1f2937; color: white; padding: 12px 30px; border-radius: 6px; text-decoration: none;">💻 View Code</a>
</div>

</div>
