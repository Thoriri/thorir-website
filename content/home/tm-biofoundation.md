---
widget: blank
headless: true
weight: 33
title: ""
design:
  columns: '1'
  spacing:
    padding: ['20px', '0', '40px', '0']
---

<div class="tm-bf">
  <div>
    <div class="tm-kicker">Flagship software</div>
    <h2>BioFoundation</h2>
    <div class="tm-bf-stats" id="tm-gh-stats"><span>★ <b data-gh="stars">121</b> stars</span><span>⑂ <b data-gh="forks">15</b> forks</span><span><b>Apache 2.0</b></span></div>
    <p>The open-source home of our biosignal foundation-model family. PyTorch Lightning, Hydra configs, pre-trained weights on Hugging Face, and distributed training: everything needed to pre-train, fine-tune, and deploy.</p>
    <div class="tm-btns">
      <a class="tm-btn tm-btn-light" href="https://github.com/pulp-bio/biofoundation" target="_blank" rel="noopener">View on GitHub</a>
      <a class="tm-btn tm-btn-dark-ghost" href="https://huggingface.co/PulpBio" target="_blank" rel="noopener">Weights on Hugging Face →</a>
    </div>
  </div>
  <div class="tm-models">
    <div class="tm-model"><img src="/uploads/model-logos/luna.png" alt="LUNA logo" loading="lazy"><span><span class="tm-mname">LUNA</span><br><span class="tm-mdesc">Topology-agnostic transformer · 300× fewer FLOPs</span></span><span class="tm-venue">NeurIPS 2025</span></div>
    <div class="tm-model"><img src="/uploads/model-logos/femba.png" alt="FEMBA logo" loading="lazy"><span><span class="tm-mname">FEMBA</span><br><span class="tm-mdesc">Bidirectional Mamba · linear-time, 0.949 AUROC</span></span><span class="tm-venue">EMBC 2025</span></div>
    <div class="tm-model"><img src="/uploads/model-logos/lumamba.png" alt="LuMamba logo" loading="lazy"><span><span class="tm-mname">LuMamba</span><br><span class="tm-mdesc">LUNA + FEMBA + LeJEPA · 377× cheaper than LaBraM</span></span><span class="tm-venue">EUSIPCO 2026</span></div>
    <div class="tm-model"><img src="/uploads/model-logos/panluna.png" alt="PanLUNA logo" loading="lazy"><span><span class="tm-mname">PanLUNA</span><br><span class="tm-mdesc">Multimodal EEG + ECG + PPG · 5.4M params</span></span><span class="tm-venue">AICAS 2026</span></div>
    <div class="tm-model"><img src="/uploads/model-logos/tinymyo.png" alt="TinyMyo logo" loading="lazy"><span><span class="tm-mname">TinyMyo</span><br><span class="tm-mdesc">3.6M-param EMG model for microcontrollers</span></span><span class="tm-venue">arXiv 2025</span></div>
    <div class="tm-model"><span class="tm-mono">CB</span><span><span class="tm-mname">CEReBrO</span><br><span class="tm-mdesc">Compact encoder · alternating attention</span></span><span class="tm-venue">arXiv 2025</span></div>
  </div>
</div>

<script>
(function(){
  fetch('https://api.github.com/repos/pulp-bio/biofoundation')
    .then(function(r){ return r.json(); })
    .then(function(d){
      if(d.stargazers_count){
        document.querySelector('[data-gh="stars"]').textContent = d.stargazers_count;
        document.querySelector('[data-gh="forks"]').textContent = d.forks_count;
      }
    }).catch(function(){});
})();
</script>
