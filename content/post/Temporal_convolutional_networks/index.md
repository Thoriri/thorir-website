---
title: Temporal Convolutional Networks
subtitle: A short introduction into how Temporal Convolutional Networks function

# Summary for listings and search engines
summary: An introduction into dilated causal convolutions, and a look into how Temporal Convolutional Networks (TCN) function.

# Link this post with a project
projects: []

# Date published
date: "2021-11-10T12:00:00Z"

# Date updated
lastmod: "2021-11-10T12:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Dilated Convolution'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- Academic

categories:
---

Temporal Convolutional Networks (TCN) which are a variaton of Convolutional Neural Networks (CNN), have been used recently by deep learning practitioners to solve time series tasks with promising and successful outcomes as seen here [CITE]. I for one have employed TCNs for detecting Arythmia in ECG signals with great success. In this short post I want to explain how these networks work, how they differ from normal CNNs and take a look into the computational workload.

For sake of illusration I will explain all of these concepts here in 1D, but they also work in higher dimensions. First let us look at normal a CNN, let's assume that we have one layer, which has a kernel size of 3 and 1 filter. And let's assume that we have a input time series that looks like the one here below:
![Time series example](uploads/time_series.PNG "Example of time series")

When we then want to apply the 1D convolution to this input time series we do the following: We take our kernel size, which is 3, and slide it over the input time series to produce a output time series. Now how does this actually look like? Let's look at the first output of the output time series and see how that is produced,
![Showing how first sample of output time seris is formed](uploads/conv.gif "Showing how first sample of output time seris is formed")
We then slide the kernel over the whole input time series and get the following output:
![Output time series](uploads/time_series.PNG "Output time series")
Now first thing we notice is that the output time series is not the same length as the input time series. This is because we do not do any padding, and we can calculate the output length by the following formula:
$$
T_{out} = T_{in} - (k-1)
$$
Where $k$ is the kernel size.

