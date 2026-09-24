---
title: "AI Epilepsy 2027 – Keynote: EEG-Based Foundation Models"

event: "5th International Conference on Artificial Intelligence in Epilepsy and Neurological Disorders (AI Epilepsy 2027)"
event_url: "https://www.aiepilepsy-neuro.com/"
tags:
  - AI Epilepsy
  - Keynote
  - Invited Talk
  - Foundation Models
  - EEG
  - Epilepsy
location: Beaver Run Resort & Conference Center, Breckenridge, Colorado, USA
address:
  street: 620 Village Rd
  city: Breckenridge
  region: Colorado
  postcode: '80424'
  country: United States

summary: Invited keynote on EEG-based foundation models at the 5th International Conference on AI in Epilepsy & Neurological Disorders in Breckenridge, Colorado.
abstract: |
  Invited keynote at **AI Epilepsy 2027**, the 5th International Conference on Artificial Intelligence in Epilepsy and Neurological Disorders, which brings clinicians, neuroscientists, engineers, and computer scientists together to move AI for epilepsy and neurology into real-world care. The talk covers **EEG-based foundation models**: models pre-trained on tens of thousands of hours of unlabeled EEG that transfer to clinical tasks such as seizure detection, abnormality detection, and artifact rejection with little labeled data. I will discuss how our models handle the practical realities of clinical EEG — heterogeneous electrode montages (LUNA, LuMamba), long recordings at linear cost (FEMBA, LuMamba), multimodal EEG/ECG/PPG signals (PanLUNA), and label-efficient self-supervised pre-training — and what it takes to run them on wearable devices for continuous, long-term epilepsy monitoring.

date: "2027-02-22T09:00:00-07:00"
date_end: "2027-02-25T18:00:00-07:00"
all_day: true

# Publish now so the upcoming talk is listed before it happens (production builds skip future-dated pages).
publishDate: "2026-09-24T00:00:00Z"

authors: []

featured: true

links:
  - icon: external-link-alt
    icon_pack: fas
    name: AI Epilepsy 2027 conference
    url: "https://www.aiepilepsy-neuro.com/"
  - icon: external-link-alt
    icon_pack: fas
    name: LUNA paper (arXiv)
    url: "https://arxiv.org/abs/2510.22257"
url_code: "https://github.com/pulp-bio/biofoundation"
url_pdf: ""
url_slides: ""
url_video: ""

slides: ""

projects: []
---

I'm honoured to have been invited to give a keynote on **EEG-based foundation models** at [AI Epilepsy 2027](https://www.aiepilepsy-neuro.com/), held **February 22–25, 2027** in Breckenridge, Colorado — the mountain town where the conference series began in 2023.

## What I'll cover

- **Why foundation models for EEG:** labeled clinical EEG is scarce and expensive, but unlabeled recordings are abundant. Self-supervised pre-training turns that archive into models that adapt to new clinical tasks with little annotation.
- **Any montage, any length:** clinical and wearable EEG use very different electrode layouts and recording durations. [LUNA](/publication/2025-neurips-luna/) (NeurIPS 2025) reads any electrode layout into a shared latent space, [FEMBA](/publication/2025-embc-femba/) and [LuMamba](/publication/2026-eusipco-lumamba/) scale linearly with recording length, and [PanLUNA](/publication/2026-aicas-panluna/) extends the idea to joint EEG, ECG, and PPG.
- **From the cloud to the patient:** compressing these models until they run in real time on ultra-low-power wearables — the path towards continuous, long-term epilepsy monitoring outside the hospital.

## Resources

- [Conference website →](https://www.aiepilepsy-neuro.com/)
- [BioFoundation codebase →](https://github.com/pulp-bio/biofoundation)
- [BioFoundation project →](/project/biofoundation/)
