<!-- # Cox Proportional Hazard models-->

<style>
  h1, h2, h3 { color: #04A9F4; }
</style>

# Semi-Parametric/Cox Proportional Hazard models
---

## [Cox Proportional Hazard model](coxph.md)

### Hazard function's formula

When it comes to predicting the survival function for a specific unit, the [Cox Proportional Hazard Model (CoxPH)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is usually the go-to model. 
The CoxPH model is a semi-parametric model that focuses on modeling the hazard function $h(t, x_i)$, by assuming that its time component $\lambda_0(t)$ and feature component $\eta(\vec{x_i})$ are proportional such that:
\begin{equation*}
h(t, \vec{x_i}) = \lambda_0(t)\eta(\vec{x_i})
\end{equation*}
with:

* $\lambda_0(t)$, is the baseline function, which is usually not specified.

* $\eta(\vec{x_i})$, is the risk function usually expressed via a linear representation such that $\eta(\vec{x_i}) = \exp \left( \sum_{j=1}^p x^i_j\omega_j \right)$. $\omega_j $ are  the coefficients to determine


### Building the model

The model can be built by calculating the Efron's partial likelihood to take ties into account. 
The partial likelihood $L(\omega)$ can be written such that:
$\displaystyle L(\omega )=\prod _{j}{\frac {\prod _{i\in H_{j}}\theta _{i}}{\prod _{s =0}^{m_j-1}[\sum _{i:T_{i}\geq t_{j}}\theta _{i}-{\frac {s}{m_j}}\sum _{i\in H_{j}}\theta _{i}]}}.$


* the log partial likelihood is $\displaystyle l (\omega )=\sum _{j}\left(\sum _{i\in H_{j}}X_{i}\cdot \omega -\sum _{s =0}^{m_j-1}\log(\phi _{j,s ,m_j})\right).$

* the gradient is $\displaystyle \vec{\nabla l(\omega )}=\sum _{j}\left(\sum _{i\in H_{j}}X_{i}-\sum _{s =0}^{m_j-1}{\frac {z_{j,s ,m_j}}{\phi _{j,s ,m_j}}}\right).$

* the Hessian matrix is $\displaystyle \nabla^2 l(\omega )=-\sum _{j}\sum _{s =0}^{m_j-1}\left({\frac {{Z}_{j,s ,m_j}}{\phi _{j,s ,m_j}}}-{\frac {z_{j,s ,m}z_{j,s ,m_j}^{T}}{\phi _{j,s ,m_j}^{2}}}\right).$


We can now use the **Newton-Optimization schema** to fit the model:
\begin{equation*}
\omega_{new} = \omega_{old} -  \nabla^2 l(\omega )^{-1} \cdot \vec{\nabla l(\omega )}
\end{equation*}

with:

* $H_j = \left\{ i; T_i = t_j \text{ and } E_i = 1 \right \}$,  ${m_j} = \left|H_j \right|.$
* $\theta _{i}= \exp\left(X_{i} \cdot \omega \right)$
* $\phi _{j,s ,m_j}=\sum _{i:T_{i}\geq t_{j}}\theta _{i}-{\frac {s}{m_j}}\sum _{i\in H_{j}}\theta _{i}$ 
* $ z_{j,s ,m_j}=\sum _{i:T_{i}\geq t_{j}}\theta _{i}X_{i}-{\frac {s }{m_j}}\sum _{i\in H_{j}}\theta _{i}X_{i}.$ 
* $ {Z}_{j,s ,m_j}=\sum _{i:T_{i}\geq t_{j}}\theta _{i}X_{i}X_{i}^{T }-{\frac {s }{m_j}}\sum _{i\in H_{j}}\theta _{i}X_{i}X_{i}^{T}$ 


---


## [DeepSurv/NonLinear CoxPH model](nonlinear_coxph.md)

### Hazard function's formula

The NonLinear CoxPH model was popularized by Katzman et al. in **[DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network](https://arxiv.org/pdf/1606.00931.pdf)** by allowing the use of Neural Networks within the original design. Here the hazard function $h(t, x_i)$ can be written as  
\begin{equation*}
h(t, \vec{x_i}) = \lambda_0(t)\Psi(\vec{x_i})
\end{equation*}

with:

* $\Psi(\vec{x_i}) = \exp(\psi(\vec{x_i}))$, where $\psi$ is a non-linear risk function.


### Building the model

We are still using the Efron's partial likelihood to take ties into account, but here the hazard function is $h(t, \vec{x_i}) = \lambda_0(t)\Psi(\vec{x_i})$. Thus, the log partial likelihood is 
\begin{equation*}
l (\omega )=\sum _{j}\left(\sum _{i\in H_{j}}\log(\Psi(\vec{x_i})) -\sum _{s =0}^{m_j-1}\log \left(\sum _{i:T_{i}\geq t_{j}}\Psi(\vec{x_i})-{\frac {s }{m_j}}\sum _{i\in H_{j}}\Psi(\vec{x_i})\right)\right).
\end{equation*}

As the Hessian matrix will be too complicated to calculate, we will use [`PyTorch`](https://pytorch.org/) to compute the gradient and perform a First-Order optimization.


---

## References
* [Wikipedia - Proportional hazards model](https://en.wikipedia.org/wiki/Proportional_hazards_model#Tied_times)
* [Cox, David R. "Regression models and life‐tables." Journal of the Royal Statistical Society: Series B (Methodological) 34.2 (1972): 187-202.](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x)
* [Katzman, Jared, et al. "DeepSurv: Personalized treatment recommender system using A Cox proportional hazards deep neural network." arXiv preprint arXiv:1606.00931 (2016).](https://arxiv.org/pdf/1606.00931.pdf)