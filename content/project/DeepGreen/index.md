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
In 2018 Joe Warrington started the Automatic Control Lab’s project to build a robotic snooker player capable of challenging a top-ranking human. The project was called DeepGreen, inspired by Deep Blue, the chess computer that famously defeated Garry Kasparov back in 1996. In the spring of 2019 I worked on a research project with Joe that combined robotics, computer vision and a little bit of machine learning.

The project had the title of "High-Precision Control and Localisation for Robotic Billiard Shots" the project aimed at optimally combining the outputs from two cameras to take accurate billiard shots.

The cue mounted camera, using image processing techniques extracts the $(x,y)$ position in pixel coordinates of each ball as seen in the figure below, the distance to each ball, $d$, is then measured using the built in depth sensor. For being able to accurately estimate the pose of the cue camera we need to have each ball coordinate in a cue camera coordinate system.

![(x, y) coordinates of three different balls.](uploads/balls.png) 

### Cue camera coordinate system
The field of view of the camera is $69.4^{\circ}$ in the horizontal direction and $49.5^{\circ}$ in the vertical direction. We make the assumption that the camera coordinate system is as seen in Figure \ref{fig:cameracord}. We can then calculate the two angles $\theta$ (horizontal FOV) and $\phi$ (vertical FOV).
For $\theta$ we divide the picture into two parts as seen in the figure beow, and we then calculate:
\begin{equation}
\theta = \frac{|x-640|}{640}\times\frac{69.4^\circ}{2}
\end{equation}
Similarly for $\phi$ we get:
\begin{equation}
\phi = \frac{|y-360|}{360}\times\frac{49.5^\circ}{2}
\end{equation}

\begin{figure}[h!]
\centering
\begin{tikzpicture}
\draw (0,0) rectangle (8,4);
\fill[red] (1,2) circle (0.5cm);
\fill[black] (1,2) circle (0.05cm);
\fill[yellow] (4,2) circle (0.5cm);
\fill[black] (4,2) circle (0.05cm);
\fill[green] (7,2) circle (0.5cm);
\fill[black] (7,2) circle (0.05cm);
\node at (1,3) {Ball 1};
\node at (4,3) {Ball 2};
\node at (7,3) {Ball 3};
\node at (4,4.5) {Cue-Camera frame};
\end{tikzpicture}
\caption{$(x,y)$ coordinates of three different balls.} \label{fig:balls}
\end{figure}
