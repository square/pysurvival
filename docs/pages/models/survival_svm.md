<!-- # SVM models-->

<style>
  h1, h2, h3 { color: #04A9F4; }
</style>

#  Survival SVM model
Instead of modeling the probability that an event will occur, we could look at Survival Analysis as a Ranking Problem.
Indeed, the idea behind formulating the survival problem as a ranking problem is that in some applications, like clinical applications, one is only interested in defining risks groups, and not the prediction of the survival time, but whether the unit has a high or low risk of experiencing the event. 

Van Belle et al. developed the [Rank Support Vector Machines (RankSVMs)](ftp://ftp.esat.kuleuven.be/SISTA/vanbelle/reports/07-70.pdf) and Polsterl et al. designed a straightforward algorithm to efficiently use the primal formulation, by computing a convex quadratic loss function, so that we can use the Newton optimization to minimize it, for [a linear approach](https://link.springer.com/chapter/10.1007/978-3-319-23525-7_15) and [a nonlinear/kernel based approach](https://arxiv.org/abs/1611.07054).

---

## [Linear approach](linear_svm.md)

The objective function of ranking-based linear survival support vector machine is defined as:
\begin{equation}
L= \frac{1}{2} || \vec{\omega} ||^2 +  \frac{\gamma}{2} \sum_{i, j \in P} \max\left(0, 1 - \vec{\omega}^T \cdot \left( \vec{x_i}- \vec{x_j} \right)  \right)^2
\end{equation}

with $ P = \{ (i, j) | T_i > T_j \text{ and } \delta_j = 1 \} $ and $p_0=|P|$

The objective function, gradient and Hessian can be expressed in matrix form as:
\begin{equation}
  \begin{split}
L(\omega) & = \frac{1}{2} \vec{\omega} \cdot \vec{\omega}^T +  \frac{\gamma}{2}\left( p_0 +  \vec{\omega}^T \cdot X^T \left(A \cdot X \cdot \vec{\omega} - 2\vec{a}\right) \right) \\
\frac{\partial L}{\partial \omega} & = \vec{\omega} + \gamma X^T \cdot \left( A \cdot  X \cdot \vec{\omega} - \vec{a} \right) \\
\frac{\partial^2 L}{\partial \omega \partial \omega^T} & = I + \gamma X^T \cdot  A \cdot  X  \\
  \end{split}
\end{equation}

with: 

* $\gamma$, the L2 regularization parameter
* $I$, the identity matrix
* $[ \vec{a} ]_i = l^{-}_{i} - l^{+}_{i} $

* $\left[ A \right]_{i,i} = l^{-}_{i} + l^{+}_{i} $ if $i=j$; $\left[ A \right]_{i,j} = -1 $ if $i\neq j$ and $j \in \text{SV}_{i}$; and $\left[ A \right]_{i,j} = 0$ otherwise. 

	* $\text{SV}_{i}^{+} = \{ s | T_s > T_i \text{ and } \omega^T x_s < \omega^T x_i  + 1 \text{ and } \delta_i = 1 \}$ and $l^{+}_{i} = |\text{SV}_{i}^{+}|$
	* $\text{SV}_{i}^{+} = \{ s | T_s < T_i \text{ and } \omega^T x_s > \omega^T x_i  - 1 \text{ and } \delta_s = 1 \}$ and $l^{-}_{i} = |\text{SV}_{i}^{-}|$
	* $\text{SV}_{i} = \text{SV}_{i}^{+} \bigcup \text{SV}_{i}^{-}$


---

## [Kernel approach](kernel_svm.md)

It is possible to model non-linearities and interactions within the covariates by using kernel-based methods. 
\begin{equation}
L = \frac{1}{2}||\phi||^2 + \frac{\gamma}{2}\sum_{i, j \in P} \max\left( 0, 1 - (\phi(x_i)-\phi(x_j) \right)^2 
\end{equation}


The objective function, gradient and Hessian can be expressed in matrix form as:
\begin{equation}
  \begin{split}
L(\beta) & = \frac{1}{2}\vec{\beta} \cdot K \cdot \vec{\beta}^T+ \frac{\gamma}{2}\left( p_0 + \vec{\beta}^T \cdot K^T \left(A \cdot K \cdot \vec{\beta}- 2\vec{a}\right) \right) \\
\frac{\partial L}{\partial \beta} & = K \cdot \vec{\beta} + \gamma K \left( A \cdot  K \cdot \vec{\beta} - \vec{a} \right) \\
\frac{\partial^2 L}{\partial \beta \partial \beta^T} & = K + \gamma K \cdot A  \cdot K  \\
  \end{split}
\end{equation}

with: 

* $\gamma$, the L2 regularization parameter

* $K$ is the $n \times n$ symmetric positive definite kernel matrix such that $\forall (i,j) \in [\![ 1, n ]\!] \times  [\![ 1, n ]\!]$, $K_{i,j} = k(x_i, x_j)$  , with $k$, a kernel function and $n$, the number of samples.
* $\vec{K_s} = \left[ k(x_s, x_1), k(x_s, x_2), ..., k(x_s, x_n)\right] $

* $[ \vec{a} ]_i = l^{-}_{i} - l^{+}_{i} $

* $\left[ A \right]_{i,i} = l^{-}_{i} + l^{+}_{i} $ if $i=j$; $\left[ A \right]_{i,j} = -1 $ if $i\neq j$ and $j \in \text{SV}_{i}$; and $\left[ A \right]_{i,j} = 0$ otherwise. 

	* $\text{SV}_{i}^{+} = \{ s | T_s > T_i \text{ and } \vec{K_s}^T \cdot \vec{\beta} <  \vec{K_i}^T \cdot \vec{\beta}  + 1 \text{ and } \delta_i = 1 \}$ and $l^{+}_{i} = |\text{SV}_{i}^{+}|$
	* $\text{SV}_{i}^{+} = \{ s | T_s < T_i \text{ and } \vec{K_s}^T \cdot \vec{\beta} >  \vec{K_i}^T \cdot \vec{\beta}  - 1 \text{ and } \delta_s = 1 \}$ and $l^{-}_{i} = |\text{SV}_{i}^{-}|$
	* $\text{SV}_{i} = \text{SV}_{i}^{+} \bigcup \text{SV}_{i}^{-}$



---

## References

* [Van Belle, Vanya, et al. "Support vector machines for survival analysis." Proceedings of the Third International Conference on Computational Intelligence in Medicine and Healthcare (CIMED2007). 2007.](ftp://ftp.esat.kuleuven.be/SISTA/vanbelle/reports/07-70.pdf)
* [Pölsterl, Sebastian, et al. "Fast training of support vector machines for survival analysis." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Cham, 2015.](https://link.springer.com/chapter/10.1007/978-3-319-23525-7_15)
* [Slides about "Fast training of support vector machines for survival analysis."](https://k-d-w.org/slides/ecml-2015/poelsterl2015ecmlpkdd.slides.pdf)
* [Pölsterl, Sebastian, et al. "An Efficient Training Algorithm for Kernel Survival Support Vector Machines." arXiv preprint arXiv:1611.07054 (2016).](https://arxiv.org/abs/1611.07054)