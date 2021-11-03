---
title: High-Precision Control and Localisation for Robotic Billiard Shots
summary: Taking successful shots on a snooker table is an hard task and requires high-precision control and localisation of the cue. We implemented a vision-based strategy that combines the fields of vision of two cameras, an overhead camera pointing down on the table and a cue mounted camera, allowing a robot arm to obtain the excellent cueing accuracy observed in human players. The robot arm holds a linear motor that plays the role of the cue and the cue camera is mounted on top of the linear motor. Cue pose estimation incorporates orthogonal uncertainty Procrustes analysis. From pose estimation and image information we make feedback corrections that optimally line up the cue so it strikes the cue ball in the desired shot angle. The feedback is then fed to a robot arm that makes the corrections. This method is then implemented and tested on a snooker table with shots of varying difficulty.
date: "2019-05-30T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: 
  focal_point: Smart

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/thorirmar
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
header-includes: |
    \usepackage{tikz,pgfplots}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhead[CO,CE]{This is fancy}
    \fancyfoot[CO,CE]{So is this}
    \fancyfoot[LE,RO]{\thepage}
---
In 2018 Joe Warrington started the Automatic Control Lab’s project to build a robotic snooker player capable of challenging a top-ranking human. The project was called DeepGreen, inspired by Deep Blue, the chess computer that famously defeated Garry Kasparov back in 1996. In the spring of 2019 I worked on a research project with Joe that combined robotics, computer vision and a little bit of machine learning.

The project had the title of "High-Precision Control and Localisation for Robotic Billiard Shots" the project aimed at optimally combining the outputs from two cameras to take accurate billiard shots.
| ![(x, y) coordinates of three different balls.](uploads/balls.png) | 
\begin{tikzpicture}
\begin{axis}
\addplot[color=red]{exp(x)};
\end{axis}
\end{tikzpicture}

