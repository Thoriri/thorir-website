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
---
This is the body of the text

Figure is below...

```latex {cmd=true hide=true}
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{matrix}
\begin{document}
\begin{tikzpicture}
  \matrix (m) [matrix of math nodes,row sep=3em,column sep=4em,minimum width=2em]
  {
     F & B \\
      & A \\};
  \path[-stealth]
    (m-1-1) edge node [above] {$\beta$} (m-1-2)
    (m-1-2) edge node [right] {$\rho$} (m-2-2)
    (m-1-1) edge node [left] {$\alpha$} (m-2-2);
\end{tikzpicture}
\end{document}

