---
title: "Research"
summary: "Machine learning that is clinically useful on devices running for weeks on a coin cell. Three directions, one goal."
share: false
reading_time: false
show_related: false
---

<div class="tm-page">

<div class="tm-rsec" id="foundation-models">
  <div>
    <h2><span class="tm-dot" style="background:#215caf"></span>Foundation models for biosignals</h2>
    <p>Every EEG dataset uses a different electrode montage, and every wearable has a different channel count. I build pre-trained models that read <b>any</b> layout into a shared latent space, so one model serves many devices and tasks with minimal fine-tuning.</p>
    <p>LUNA introduced query-based channel unification at NeurIPS 2025. FEMBA swapped quadratic attention for linear-time Mamba. LuMamba combines both and adds LeJEPA world-model pre-training. TinyMyo extends the family to EMG. All are open source in <a href="https://github.com/pulp-bio/biofoundation">BioFoundation</a> with weights on Hugging Face.</p>
    <div class="tm-chipset">
      <a class="tm-chip" href="/publication/2025-neurips-luna/">LUNA · NeurIPS 2025</a>
      <a class="tm-chip" href="/publication/2025-embc-femba/">FEMBA · EMBC 2025</a>
      <a class="tm-chip" href="/post/luna_eeg_foundation_model/">Blog: LUNA explained</a>
      <a class="tm-chip" href="/post/lejepa_eeg_pretraining/">Blog: LuMamba pre-training</a>
    </div>
  </div>
  <div class="tm-sidecard">
    <h4>Key numbers</h4>
    <ul>
      <li><span class="tm-v">300×</span> fewer FLOPs than standard transformers (LUNA)</li>
      <li><span class="tm-v">21,000+ h</span> of EEG in pre-training</li>
      <li><span class="tm-v">0.97 AUPR</span> Alzheimer's detection on unseen montages (LuMamba)</li>
    </ul>
  </div>
</div>

<div class="tm-rsec" id="tiny-recursion-models">
  <div>
    <h2><span class="tm-dot" style="background:#7c3aed"></span>Tiny recursion models</h2>
    <p>Reasoning-style computation does not have to mean billions of parameters. I study how deep supervision and adaptive-depth recursion let very small networks refine their answers iteratively, spending compute only where the signal demands it.</p>
    <p>Recent work reframed TRM recursion as annealed sampling on an energy-based model, running the loop on a thermodynamic-computing simulator. Two open MSc topics extend TRMs to time series and quantized edge deployment.</p>
    <div class="tm-chipset">
      <a class="tm-chip" href="/post/thermo-trm-thermodynamic-reasoning/">Blog: Thermo-TRM</a>
      <a class="tm-chip" href="/project/trm-timeseries/">MSc topic: TRMs for time series</a>
      <a class="tm-chip" href="/project/trm-quantized/">MSc topic: Quantized TRMs</a>
    </div>
  </div>
  <div class="tm-sidecard">
    <h4>Key numbers</h4>
    <ul>
      <li><span class="tm-v">7×</span> fewer sampling sweeps via recursion (Thermo-TRM)</li>
      <li><span class="tm-v">2nd place</span> ETH Probabilistic Computing Hackathon</li>
    </ul>
  </div>
</div>

<div class="tm-rsec" id="ultra-low-power-deployment">
  <div>
    <h2><span class="tm-dot" style="background:#0d9488"></span>Ultra-low-power deployment</h2>
    <p>A model is only clinically useful if it runs where the patient is. I co-design algorithms and implementations for GAP9 and RISC-V platforms: quantization, on-device continual learning, and energy-aware architectures for seizure detection, BMI, and speech imagery.</p>
    <p>This line of work goes back to my PhD (EEG-TCNet, EpiDeNet, BioGAP) and continues with quantized foundation models running on wearables.</p>
    <div class="tm-chipset">
      <a class="tm-chip" href="/publication/2025-tbcas-speech-imagery/">Speech imagery · TBioCAS 2025</a>
      <a class="tm-chip" href="/publication/eeg-tcnet/">EEG-TCNet · SMC 2020</a>
      <a class="tm-chip" href="/project/droneflight/">Project: EEG drone control</a>
    </div>
  </div>
  <div class="tm-sidecard">
    <h4>Key numbers</h4>
    <ul>
      <li><span class="tm-v">µW-scale</span> inference on GAP9</li>
      <li><span class="tm-v">SOTA</span> wearable seizure detection benchmarks</li>
      <li><span class="tm-v">BCI Award</span> nomination for drone-control headband</li>
    </ul>
  </div>
</div>

</div>
