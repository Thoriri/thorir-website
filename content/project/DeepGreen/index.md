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
### Orthogonal Procrustes Problem
Now let 
$$
A_i = (x_i,y_i,0)^T, \ \ i = 1,2,\ldots,N
$$
represent the ball position in the world coordinate system. And the corresponding points as seen in the camera coordinate system let them be represented with: 
$$
B_i = (\tilde x_i,\tilde y_i,\tilde z_i)^T, \ \ i = 1,2,\ldots,N
$$
Where $N$ denotes the total number of ball seen.
We then should be able to write: 
$$
A_i = RB_i + T
$$
Where $R$ is a 3x3 rotation matrix and $T$ is a 3x1 translation matrix. This kind of problem falls under so called Procrustes analysis, and since we have the necessary condition of $R$ having to be a valid rotation matrix the problem is better known as the Orthogonal Procrustes problem. The name Procrustes refers to a bandit from Greek mythology who made his victims fit his bed by either stretching their limbs or cutting them off.
Based on work presented in \cite{lourenco_uncertainty_2017} we were able to include uncertainty in our minimization.
To find $R$ and $T$ we minimize:
\begin{equation}
\label{eqn:minimization}
\min_{R,T} \sum_{i=1}^N \sigma_i^{-2}||A_i - RB_i - T||^2
\end{equation}
Where $\sigma_i^{-2}, \ \ i = 1,2,\ldots,N $ accounts for the intrinsic uncertainty of each point pair, $A_i$ and $B_i$. Also define:
$$
\boldsymbol{\Sigma_e} = 
  \begin{pmatrix}\sigma_1^{2} & 0 & \dots & 0 \\\ 0 & \sigma_2^{2} & \dots & 0 \\\ \vdots & \vdots & \ddots & \vdots \\\ 0 & 0 & \dots & \sigma_N^{2} \end{pmatrix}
$$

$\boldsymbol{\Sigma_e}$ is a diagonal matrix that models the point uncertainty. Now since the true $\boldsymbol{\Sigma_e}$ is not known we can conservatively define the diagonal values of $\boldsymbol{\Sigma_e}$ as follows:
$$
\sigma_{i}^{2}=\lambda_{\max }\left(\boldsymbol{\Sigma}_{A_{i}}\right)+\lambda_{\max }\left(\boldsymbol{\Sigma}_{B_{i}}\right)
$$
Where $\lambda_{\max}$ denotes the maximum eigenvalue and $$
\boldsymbol{\Sigma}_{A_i} = 
  \begin{pmatrix} \delta \hat{x} & 0  & 0 \\\ 0 & \delta \hat{y}  & 0 \\\ 0 & 0 &  \delta \hat{z} \end{pmatrix}, \ 
\boldsymbol{\Sigma}_{B_i} = 
  \begin{pmatrix} \delta \tilde x & 0  & 0 \\\ 0 & \delta \tilde y  & 0 \\\ 0 & 0 &  \delta \tilde z \end{pmatrix}
$$
denote the uncertanty matrices of points detected by the overhead camera and cue camera respectively.
By taking the derivative with regard to $T$ and setting equal to zero we get:
$$
\sum_{i=1}^N\sigma_{i}^{-2}(A_i - RB_i - T) \overset{!}{=} 0 \
$$
$$
\rightarrow T = \frac{1}{N_{\sigma}}\sum_{i=0}^{N} \sigma_i^{-2}A_i - \frac{R}{N_{\sigma}}\sum_{i=0}^{N} \sigma_i^{-2}B_i
$$
or 
$$
T = \frac{1}{N_\sigma}(\bar{A}\boldsymbol{\Sigma_e^{-1}}\unicode{x1D7D9}- R\bar{B}\boldsymbol{\Sigma_e^{-1}}\unicode{x1D7D9})
$$
or
$$
T = \mu_{A} - R \mu_{B}
$$
where: $\bar{A} = [A_1,\ldots,A_N]$, $\bar{B} = [B_1,\ldots,B_N]$, $N_\sigma = \sum_{i=1}^{N}\sigma_i^{-2}$, $\unicode{x1D7D9} = [1,1,\ldots,1]^T \in \mathbb{R}^{N} $, $\mu_{A} = \frac{1}{N_\sigma}\bar{A}\boldsymbol{\Sigma_e^{-1}}\unicode{x1D7D9} $ and $\mu_{B} = \frac{1}{N_\sigma}\bar{B}\boldsymbol{\Sigma_e^{-1}}\unicode{x1D7D9}$

We can then rewrite EQUATION to be

$$ \begin{align} & \underset{R}{\text{min}} & &  ||A - RB||_F^2\\\ & \text{subject to} & & R^TR = I_3 \ \ \text{and}  \ \det{(R)} = 1   \end{align} $$

where $A = \tilde{A}\boldsymbol{\Sigma_e^{-1/2}}$, $B = \tilde{B}\boldsymbol{\Sigma_e^{-1/2}}$, $\tilde{A} = (A_1 - \mu_A,A_2 - \mu_A,\ldots,A_N - \mu_A)$ and $\tilde{B} = (B_1 - \mu_{B},B_2 - \mu_{B},\ldots,B_N - \mu_{B})$.
Now we can write:
$$ 
\|A-RB\|_{F}^{2}=\operatorname{trace}\left(A^{T} A-2 B^{T}R^{T} A+B^{T} B\right)
$$
So the minimization problem \eqref{eqn:minimization_rewritten} can be thought of as maximizing $\operatorname{trace}\left(R B A^{T}\right)$  since $\operatorname{trace}\left(B^{T}R^{T} A\right) = \operatorname{trace}\left(R B A^{T}\right)$ by properties of the trace.
Now let $BA^{T} = C$ and then:

$$ \begin{align} \operatorname{trace}\left(R C\right) & =\operatorname{trace}\left(R U \Sigma V^{T}\right) \\\ & =\operatorname{trace}\left(V^{T} R U \Sigma\right) \\\ & =\operatorname{trace}(Z \Sigma)\end{align} $$

Where the singular value decomposition of $C$ is $U \Sigma V^T$ and $\Sigma = \operatorname{diag}(\hat{\sigma}_1,\hat{\sigma}_2,\ldots,\hat{\sigma}_N)$.
To maximize $\operatorname{trace}\left(R B A^{T}\right)$ we must look at two cases, when $\operatorname{det}(VU^T) = \pm 1$. Note that the determinant of $VU^T$ can never be anything but $\pm 1$ since $V$ and $U$ are both orthogonal matrices and the product of two orthogonal matrices is also orthogonal. Based on algebraic proofs presented in \cite{lawrence_purely_2019} we are able to go through the following justification for each case.
Firstly we note that $\operatorname{trace}(Z \Sigma) =\sum_{i=1}^{N} z_{i i}\hat{\sigma}_{i} \leq \sum_{i=1}^{N} \ \hat{\sigma}_{i}$ since $Z$ is orthogonal and $z_{i i} \leq 1, \ i = 1,2,\ldots,N$.
If $\operatorname{det}(VU^T) = -1$, then let $D$ be a $N\times N$ orthogonal matrix so that: $\overline{Z}=D^{T} Z D,$ where $\overline{Z}=\left( \begin{array}{cc}{Z_{0}} & {O} \\ {O^{T}} & {-1}\end{array}\right)$ where $Z_{0}$ is the uppermost $N - 1 \times N - 1$ entries of $\overline{Z}$ and $O$ is a vertical column or vector of $N-1$ zeros. Then let $S=D^{T} \Sigma D = \left( \begin{array}{cc}{S_{0}} & {a} \\ {b^{T}} & {\gamma}\end{array}\right)$ where $S_0$ is the uppermost $N - 1 \times N - 1$ entries of $S$. $a$ and $b$ are vertical columns of $N-1$ entries and $\gamma$ is a scalar.\\\\
Then note:
$$ 
\operatorname{trace}(Z \Sigma)=\operatorname{trace}\left(D^{T} Z \Sigma D\right)=\operatorname{trace}\left(D^{T} Z D D^{T} \Sigma D\right)=\operatorname{trace}(\overline{Z}S) = \operatorname{trace}(Z_0S_0) - \gamma
 $$
We can also write: $$\operatorname{trace}(Z_0S_0) \leq \operatorname{trace}(S_0)$$ 
where of course: $\operatorname{trace}(S_0) + \gamma = \operatorname{trace}(S) = \operatorname{trace}(\Sigma)$.
Now we then have that $\gamma = \sum_{i=1}^{N} d_{iN}^2\hat{\sigma}_i$ so we can write with certainty that $\gamma \geq \hat{\sigma}_N$.\\
Using all these justifications we can get an upper bound on $\operatorname{trace}(Z \Sigma)$:
$$ 
\operatorname{trace}(Z \Sigma)=\operatorname{trace}\left(Z_{0} S_{0}\right)-\gamma \leq \operatorname{trace}\left(S_{0}\right)-\gamma=\operatorname{trace}(\Sigma)-\gamma-\gamma \leq \sum_{i=1}^{N-1} \hat{\sigma}_{i}-\hat{\sigma}_{N}
$$
These are then the two cases we must consider:
\begin{itemize}
\item When $\operatorname{det}(VU^T) = 1$\\
We maximize by choosing $Z = I_N$ so $R = VU^T$ minimizes \eqref{eqn:minimization_rewritten}.
\item When $\operatorname{det}(VU^T) = -1$\\
Then define $\tilde{S} = \operatorname{diag}(1,1,\ldots,-1)$ a $N \times N$ diagonal matrix. And we know that:
$$ 
\operatorname{trace}(R C)=\operatorname{trace}\left(R U \Sigma V^{T}\right)=\operatorname{trace}\left(V^{T} R U \Sigma \right) \leq \sum_{j=1}^{N-1} \hat{\sigma}_{j}-\hat{\sigma}_{N}
 $$
If we choose $R = V\tilde{S}U^T$, then $R$ is orthogonal and $\operatorname{det}(R) = 1$ and 
$$ 
\operatorname{trace}(R C)=\operatorname{trace}\left(V\tilde{S}U^TU\Sigma V^T\right)=\operatorname{trace}\left(V \tilde{S} \Sigma V^T \right) = \operatorname{trace}(\tilde{S}\Sigma) = \sum_{j=1}^{N-1} \hat{\sigma}_{j}-\hat{\sigma}_{N}
 $$
Thus $R = V\tilde{S}U^T$ minimizes \eqref{eqn:minimization_rewritten}.
\end{itemize}
\section{Euler angles}
Having estimated the rotation matrix we need to extract the Euler angles since we use them to correct the cue angle. Now let the rotation matrix be:
$$
R=\left[ \begin{array}{lll}{R_{11}} & {R_{12}} & {R_{13}} \\ {R_{21}} & {R_{22}} & {R_{23}} \\ {R_{31}} & {R_{32}} & {R_{33}}\end{array}\right]
$$
Based on work presented in \cite{slabaugh_computing_nodate} we can then calculate the Euler angles as:
\begin{algorithm}
\begin{algorithmic}
\If {$R_{31} \neq \pm 1$}
    \State $\theta \gets -\operatorname{arcsin}\left(R_{31}\right)$
    \State $\psi \gets \arctantwo \left(\frac{R_{32}}{\cos{\theta}},\frac{R_{33}}{\cos{\theta}}\right)$
    \State $\phi \gets \arctantwo \left(\frac{R_{21}}{\cos{\theta}},\frac{R_{11}}{\cos{\theta}}\right)$
\Else
    \State $\phi \gets 0$
    \If {$R_{31} = -1$}
        \State $\theta \gets \frac{\pi}{2}$
	\State $\psi \gets \arctantwo (R_{12},R_{13})$
    \Else
        \State $\theta \gets -\frac{\pi}{2}$
	\State $\psi \gets \arctantwo (-R_{12}-,R_{13})$	
    \EndIf
\EndIf
\end{algorithmic}
 \caption{Euler angle calculations from rotation matrix.}
\end{algorithm}
The most important angle is $\theta$ as that is the one can be thought of as the cue angle on the table.

