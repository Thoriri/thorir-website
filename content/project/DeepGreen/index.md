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

![(x, y) coordinates of three different balls.\label{mylabel}](uploads/balls.png) 

### Cue camera coordinate system
The field of view of the camera is $69.4^{\circ}$ in the horizontal direction and $49.5^{\circ}$ in the vertical direction. We make the assumption that the camera coordinate system is as seen in Figure \ref{mylabel}. We can then calculate the two angles $\theta$ (horizontal FOV) and $\phi$ (vertical FOV).
For $\theta$ we divide the picture into two parts as seen in the figure beow, and we then calculate:
\begin{equation}
\theta = \frac{|x-640|}{640}\times\frac{69.4^\circ}{2}
\end{equation}
Similarly for $\phi$ we get:
\begin{equation}
\phi = \frac{|y-360|}{360}\times\frac{49.5^\circ}{2}
\end{equation}

![Coordinate system of the cue mounted camera.](uploads/cue_camera_frame.png) 
![Dividing camera frame into two parts vertically.](uploads/div.png) 
![Dividing camera frame into two parts horizontally.](uploads/divv.png) 
Having extracted these two angles we can find the $(x,y,z)$ coordinates of the balls in the camera coordinate system.
The $z$-coordinate of the balls is calculated as:
\begin{equation}
\label{eqn:z_camera}
z = -\sin{(\phi)}d
\end{equation}
As can be seen in Figure \ref{fig:zcoardinate}.
![Trigonometry for finding z-coordinate.](uploads/1.png) 
The negative sign is because we assume that the camera is always above the balls.
For the $(x,y)$ coordinates we calculate the projection of $d$ onto the $x-y$ plane which is given as: 
\begin{equation}
\label{eqn:d_prime}
d' = \cos{(\phi)}d
\end{equation}
The $(x,y)$ coordinate is then calculated as can be seen from Figure \ref{fig:xycoardinate}.
![Trigonometry for finding (x; y)-coordinate.](uploads/2.png)
\begin{equation}
\label{eqn:x_y_camera}
x = \sin{(\theta)}d', \ \ y = \cos{(\theta)}d'
\end{equation}
Note if the pixel coordinate of the center of the ball is in the left half plane of the frame then: 
$$
x = -\sin{(\theta)}d'
$$
If we have at least two balls in the camera frame we can estimate the pose of the cue mounted camera.