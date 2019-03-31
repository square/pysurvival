<!-- # Simulation models-->

<style>
  h1, h2, h3 { color: #04A9F4; }
</style>

# Generating random survival times

Simulation studies represent an important statistical tool to investigate the performance, properties and adequacy of statistical models. 
Here, we will see how to generate random survival times based on the most commonly used distributions:

* Exponential
* Weibull
* Gompertz
* Log-Logistic
* Lognormal

---

## Distribution function of the Cox model 

Thanks to the  Cox proportional hazard model, it is convenient to model survival times through the hazard
function, with $h_0(t)$ the baseline function:
\begin{equation*}
h(t, x_i) = h_0(t)\exp \left( \vec{x_i} \cdot \vec{\omega} \right)
\end{equation*}


The survival function of the Cox proportional hazards models given by
\begin{equation*}
S(t, x_i) = \exp \left( - H_0(t) \cdot \exp \left( \vec{x_i} \cdot \vec{\omega} \right) \right)
\end{equation*}
with $H_0(t) = \int_0^t h(u) du $

And thus, the distribution function of the Cox model is
\begin{equation*}
F(t, x_i) = 1 - \exp \left( - H_0(t) \cdot \exp \left( \vec{x_i} \cdot \vec{\omega} \right) \right)
\end{equation*}

---

## Random survival times formula

Let $Y$ be a random variable with distribution function $F$, then $U =F(Y )$ follows a **uniform distribution** on the interval $[0,1]$, abbreviated as $U \sim Uni[0,1]$. Moreover, if $U \sim Uni[0,1]$, then $(1-U) \sim Uni[0,1]$, too. 
Thus, $U =  \exp \left( - H_0(t) \cdot \exp \left( \vec{x_i} \cdot \vec{\omega} \right) \right) \sim Uni[0,1]$

Therefore, the survival time $T$ of the Cox model can be expressed as
\begin{equation*}
T_i =  H_0^{-1} \left[ -\frac{\log(U)}{\lambda_i} \right] 
\end{equation*}
with: $\lambda_i  =  \alpha \exp\left( \vec{x_i} \cdot \vec{\omega} \right) $,  $U \sim Uni[0,1]$ and $Z \sim Normal(0, 1)$  

Therefore, as long as it is possible to compute $H_0^{-1}$, we can generate random survival times. 

* `Exponential`: $\displaystyle T_i =  -\frac{\log(U)}{\lambda_i}$
* `Weibull`    : $\displaystyle T_i =  \left(-\frac{\log(U)}{\lambda_i}\right)^{1/\beta}$
* `Gompertz`   : $\displaystyle T_i =  \frac{1}{\beta} \log\left(1-\beta \frac{\log(U)}{\lambda_i} \right)$
* `Log-Logistic` : $\displaystyle T_i = \frac{1}{\lambda_i}\left( \frac{U}{1-U} \right)^{1/\beta}$
* `Log-Normal`   : $\displaystyle T_i = \lambda_i \exp(\beta Z)$

$\alpha$ and $\beta$ are tuning parameters.


## Linear and Nonlinear hazard function
It is possible to use nonlinear hazard functions to generate random survival times such that:
\begin{equation*}
h(t, x_i) = h_0(t)\exp \left( \psi\left( \vec{x_i} \cdot \vec{\omega} \right) \right)
\end{equation*}
where $\psi$ is a nonlinear function.


## References
* [Bender, R., Augustin, T., & Blettner, M. (2005). Generating survival times to simulate Cox proportional hazards models. Statistics in medicine, 24(11), 1713-1723.](https://www.ncbi.nlm.nih.gov/pubmed/22763916)
