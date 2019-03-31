<!-- # Parametric models-->
<style>
  h1, h2, h3, h4 { color: #04A9F4; }
</style>

# Parametric models

## Introduction

We've seen that [with Semi-Parametric models](semi_parametric.md) the time component of the hazard function is left unspecified. In case the hazard function or the Survival function are known to follow or closely approximate a known distribution, it is better to use **Parametric models**.

Unlike Semi-Parametric models, Parametric models are better suited for forecasting and will return smooth functions of $h(t, x)$ or $S(t, x)$. The most common parametric models are:

* Exponential
* Weibull
* Gompertz
* Log-Logistic
* Lognormal

---

## Models

### Exponential
The exponential distribution is the simplest and most important distribution in survival studies. Being independent of prior information, it is known as a "lack of memory" distribution requiring that the present age of the living organism does not influence its future survival. 
In this model, the hazard rate is constant over time such as:
\begin{equation*}
\begin{split}
\lambda    & = \alpha  e^{ \vec{ x_i} \cdot \vec{ \omega} }\\
h(t, x_i)  & =  \lambda  \\
S(t, x_i)  & =  e^{- \lambda \cdot t} \\
\end{split}
\end{equation*}
with $\alpha$ and $\vec{\omega}$ the coefficients to find.

</b>

### Weibull
The Weibull distribution is a generalized form of the exponential distribution and is de facto more flexible than the exponential model.
\begin{equation*}
\begin{split}
\lambda    & = \alpha  e^{ \vec{ x_i} \cdot \vec{ \omega} }\\
h(t, x_i)  & =  \lambda \beta (\lambda t)^{\beta-1} \\
S(t, x_i)  & =  e^{- (\lambda t)^\beta} \\
\end{split}
\end{equation*}
with $\alpha$, $\beta$ and $\vec{\omega}$ the coefficients to find.

</b>

### Gompertz
The Gompertz distribution is a continuous probability distribution, that has an exponentially increasing failure rate, and is often applied to analyze survival data.
\begin{equation*}
\begin{split}
\lambda    & = \alpha  e^{ \vec{ x_i} \cdot \vec{ \omega} }\\
h(t, x_i)  & = \lambda e^{\beta t} \\
S(t, x_i)  & = e^{-\frac{\lambda}{\beta }(e^{\beta t}-1)}   \\
\end{split}
\end{equation*}
with $\alpha$, $\beta$ and $\vec{\omega}$ the coefficients to find.

</b>

### Log-Logistic
As the name suggests, the log-logistic distribution is the distribution of a variable whose logarithm has the logistic distribution.
\begin{equation*}
\begin{split}
\lambda    & = \alpha  e^{ \vec{ x_i} \cdot \vec{ \omega} }\\
h(t, x_i)  & = \frac{\beta\lambda^\beta t^{\beta-1}}{1+(\lambda t)^\beta} \\
S(t, x_i)  & = \frac{1}{1+(\lambda t)^\beta}   \\
\end{split}
\end{equation*}
with $\alpha$, $\beta$ and $\vec{\omega}$ the coefficients to find.


</b>

### Log-Normal
The lognormal distribution is used to model continuous random quantities when the distribution is believed to be skewed, such as lifetime variables 
\begin{equation*}
\begin{split}
\lambda    & = \alpha  e^{ \vec{ x_i} \cdot \vec{ \omega} }\\
h(t, x_i)  & =  \frac{\frac{1}{t \beta \sqrt{2\pi}} \exp \left( -\left[ \frac{\log(t) - \log(\lambda)}{\beta \sqrt{2}} \right]^2 \right)}{1 - \Phi \left( \frac{\log(t) - \log(\lambda) }{\beta}\right)}       \\
S(t, x_i)  & = 1 - \Phi \left( \frac{\log(t) - \log(\lambda) }{\beta}\right)  \\
\end{split}
\end{equation*}
with $\alpha$, $\beta$ and $\vec{\omega}$ the coefficients to find; $\Phi$ is the standard normal cdf.


</b>
</b>

---

## Building and selecting models

### Likelihood
All the parametric models will be fitted using the maximum likelihood estimation (MLE). Regardless of the model used, the likelihood is the product over all of the observations such that:
\begin{equation*}
\begin{split}
L  & =  \prod_{i=1}^{N} f(T_i, x_i)^{\delta_i} S(T_i, x_i)^{1-\delta_i} \\
   & =  \prod_{i=1}^{N} h(T_i, x_i)^{\delta_i} S(T_i, x_i) \\
\end{split}
\end{equation*}

### Selecting the best model
To select the best model, we might use the Akaike’s Information Criterion (AIC) to distinguish between different parametric models. Typically, we will pick the model whose log-likelihood is the smallest. Akaike’s
method penalizes each model’s log likelihood, $\log(L)$, to reflect the number of parameters that are being estimated and then compares them:
\begin{equation*}
AIC = −2 \log(L) + 2*\text{num_coefficients}
\end{equation*}

---

## References
* [Princeton Lecture - Parametric Survival Models](https://data.princeton.edu/pop509/ParametricSurvival.pdf)
* [UCSD Lecture - Parametric Survival Models](http://www.math.ucsd.edu/~rxu/math284/slect4.pdf)
* [The Log-Logistic Distribution](http://www.randomservices.org/random/special/LogLogistic.html)
* [The Log-Normal Distribution](http://www.randomservices.org/random/special/LogNormal.html)