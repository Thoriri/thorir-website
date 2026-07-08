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
  <svg class="tm-bf-trace" viewBox="0 0 1200 64" preserveAspectRatio="none" aria-hidden="true" focusable="false">
    <defs>
      <linearGradient id="tmBfGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#7fb2ff"/>
        <stop offset="1" stop-color="#4fd1c5"/>
      </linearGradient>
    </defs>
    <path class="tm-bf-trace-base" pathLength="1000" d="M0,32 C10,26 20,38 30,32 S45,28 58,32 Q68,20 78,32 L98,32 L108,6 L118,58 L128,32 C140,24 152,20 166,32 C180,26 192,38 204,30 S220,26 232,34 S248,26 260,34 S276,28 288,33 S304,27 316,33 S332,29 344,32 C360,28 372,34 384,32 S410,28 430,32 S460,30 480,33 S510,28 540,32 L600,32 C610,26 620,38 630,32 S645,28 658,32 Q668,20 678,32 L698,32 L708,6 L718,58 L728,32 C740,24 752,20 766,32 C780,26 792,38 804,30 S820,26 832,34 S848,26 860,34 S876,28 888,33 S904,27 916,33 S932,29 944,32 C960,28 972,34 984,32 S1010,28 1030,32 S1060,30 1080,33 S1110,28 1140,32 L1200,32"/>
    <path class="tm-bf-trace-sweep" pathLength="1000" d="M0,32 C10,26 20,38 30,32 S45,28 58,32 Q68,20 78,32 L98,32 L108,6 L118,58 L128,32 C140,24 152,20 166,32 C180,26 192,38 204,30 S220,26 232,34 S248,26 260,34 S276,28 288,33 S304,27 316,33 S332,29 344,32 C360,28 372,34 384,32 S410,28 430,32 S460,30 480,33 S510,28 540,32 L600,32 C610,26 620,38 630,32 S645,28 658,32 Q668,20 678,32 L698,32 L708,6 L718,58 L728,32 C740,24 752,20 766,32 C780,26 792,38 804,30 S820,26 832,34 S848,26 860,34 S876,28 888,33 S904,27 916,33 S932,29 944,32 C960,28 972,34 984,32 S1010,28 1030,32 S1060,30 1080,33 S1110,28 1140,32 L1200,32"/>
  </svg>
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
