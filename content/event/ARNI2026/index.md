---
title: "ARNI Working Group 2026 – Foundation Models for Wearables: From EEG and EMG to Multimodal Models at the Edge"

event: "ARNI working group: Frontier Models for Neuroscience and Behavior"
event_url: "https://arni-institute.org/events-workinggroupspeakers/"
tags:
  - ARNI
  - Invited Talk
  - Foundation Models
  - Biosignals
  - EEG
  - EMG
  - Edge Deployment
  - TinyML
location: "Online · ARNI, NSF AI Institute for Artificial and Natural Intelligence (Columbia University / University of Pennsylvania)"

summary: Invited talk at ARNI's Frontier Models for Neuroscience and Behavior working group on foundation models for wearable EEG, EMG and multimodal biosignals, from pre-training objectives to measured inference on a milliwatt-class microcontroller.
abstract: |
  We presented an overview of the biosignal foundation models developed at ETH Zürich for wearables, organised around three constraints the device imposes on the model: electrode layouts that differ between devices, recordings that run for hours, and inference that has to happen locally within a power budget of tens of milliwatts. On the EEG side we covered **LUNA**, whose learned queries read any electrode montage into a fixed-size representation, and **FEMBA**, a bidirectional state-space encoder whose cost grows linearly with recording length, and showed how **LuMamba** combines the two with 26.5× fewer FLOPs than LUNA at the longest window. We then compared pre-training objectives on LuMamba, masked reconstruction against latent-space prediction with LeJEPA, and showed that the combined objective transferred best to an unseen 16-channel montage (APAVA, AUPR 0.97 against 0.77 for reconstruction alone). For EMG we presented **TinyMyo**, a 3.6M-parameter encoder pre-trained on public surface-EMG data that adapts to hand-gesture classification (88% on NinaPro DB5), wristband gesture decoding, hand-kinematics regression (8.8° MAE on NinaPro DB8) and silent-speech decoding (33.5% WER), together with its 1.9M-parameter variant that runs in 89 ms on GAP9. **PanLUNA** extended the same query mechanism to joint EEG, ECG and PPG modelling with one 5.4M-parameter encoder that matches or exceeds specialist models up to 57× larger. The last part of the talk covered what it takes to run these models on a wearable processor: none of the INT8 encoders fits the 1.5 MB on-chip memory, quantisation-aware training recovers accuracy down to 2-bit weights, and measured latencies on GAP9 range from 89 ms to 1.3 s per window. All models and code are open: the BioFoundation models have passed 40,000 downloads on Hugging Face.

date: "2026-10-19T15:00:00-04:00"
all_day: false

publishDate: "2026-10-20T00:00:00Z"

authors: []

featured: true

image:
  caption: ""
  focal_point: Smart
  filename: featured.jpg
  preview_only: false

links:
  - icon: external-link-alt
    icon_pack: fas
    name: Models (Hugging Face)
    url: "https://huggingface.co/PulpBio"
  - icon: external-link-alt
    icon_pack: fas
    name: ARNI event page
    url: "https://arni-institute.org/events-workinggroupspeakers/"
url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: ""
url_slides: "uploads/arni2026_biosignal_fm_wearables_slides.pdf"
url_video: ""

slides: ""

projects: []
---

An invited talk at the monthly meeting of the "Frontier Models for Neuroscience and Behavior" working group of [ARNI](https://arni-institute.org/), the NSF AI Institute for Artificial and Natural Intelligence led by Columbia University. The session was held online on **19 October 2026** (15:00 US Eastern / 21:00 Zürich) and hosted by Alexandre André of Eva Dyer's lab at UPenn.

## Talk highlights

- **Three constraints, one research path:** electrode layouts, long recordings and local inference as the design inputs for every model in the family.
- **Objective matters for transfer:** masked reconstruction plus LeJEPA latent prediction gave the best transfer to an unseen montage, with the lowest variance across seeds.
- **One EMG encoder, four tasks:** TinyMyo covers gestures, kinematics and speech from a single 3.6M-parameter pre-trained model, with a 1.9M-parameter variant under 100 ms on a microcontroller.
- **Measured on the chip:** every deployment number is a measurement on GAP9 at 370 MHz, including the finding that the model with the fewest FLOPs (FEMBA) has the longest latency.

## From the slides

{{< figure src="slide02_where_we_come_from.jpg" caption="Where we come from: chips become wearables, wearables produce data, and the models go back onto the chip." >}}

{{< figure src="slide06_model_family.jpg" caption="The model family: each model answers one of the three constraints, and GAP9 is where all of them have to run." >}}

## Slides & resources

- [Download the slides (PDF) →](/uploads/arni2026_biosignal_fm_wearables_slides.pdf)
- [BioFoundation codebase →](https://github.com/pulp-bio/biofoundation)
- [Model weights on Hugging Face →](https://huggingface.co/PulpBio)
- [ARNI working group speakers page →](https://arni-institute.org/events-workinggroupspeakers/)
- [NeurIPS 2026 EEG/EMG Foundation Challenge →](https://neural-interfaces26.github.io) (mentioned in the talk; I co-organise the BCI track)

## Related work

- [LUNA (NeurIPS 2025) →](/publication/2025-neurips-luna/)
- [FEMBA (EMBC 2025) →](/publication/2025-embc-femba/)
- [LuMamba (EUSIPCO 2026) →](/publication/2026-eusipco-lumamba/)
- [PanLUNA (AICAS 2026) →](/publication/2026-aicas-panluna/)
- [TinyMyo →](/publication/2025-arxiv-tinymyo/)
- [CEReBrO →](/publication/2025-arxiv-cerebro/)
- [BioFoundation project →](/project/biofoundation/)
- [AMLD Intelligence Summit 2026 talk →](/talk/amld-intelligence-summit-2026-pushing-biosignal-fms-to-the-edge/)
