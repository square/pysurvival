<!-- # Non-Parametric models-->
<style>
  h1, h2, h3 { color: #04A9F4; }
</style>

# Non Parametric models

Non Parametric models offer a straightforward and easy-to-interpret way to compute the survival and hazard functions without imposing any assumptions.
Pysurvival provides the following non-parametric models:

* Kaplan-Meier model ([`KaplanMeierModel`](kaplan_meier.md))
* Smooth Kaplan-Meier model  ([`SmoothKaplanMeierModel`](smooth_kaplan_meier.md))


---


## [Kaplan-Meier model](kaplan_meier.md)

One of the most straight-forward ways to estimate the Survival function $S(t)$ of an entire group, is by using the **Kaplan-Meier** method. Given $N$ units in a cohort, let's assume that there are $J$ distinct actual event times such that $t_1 < t_2 < ... < t_J$ with $J \leq N$, then the Survival function estimator $S_{KM}(t)$ is given by:
\begin{equation*}
S_{KM}(t) = \prod_{t_j \leq t} \left(1-\frac{d_j}{r_j} \right) 
\end{equation*}
with:

* $S_{KM}(0) = 1$
* $d_j$ is the number of individuals experiencing an event at $t_j$
* $r_j$ is the number of individuals at risk within $[t_{j-1}, t_j)$ - *those who have not been censored or experienced an event*


---

## [Smooth Kaplan-Meier](smooth_kaplan_meier.md)

Despite its ease of use, the main drawback of the Kaplan-Meier estimator is that it is a step function with jumps. Kernel smoothing can therefore solve this issue, provided that the best kernel and bandwidth are properly chosen.

Let $S_b(t)$ be a Smooth estimator of the Kaplan-Meier survival function. $S_b(t)$ can be written such that:
\begin{equation*}
S_{b}(t) = \sum_j s_j K\left( \frac{t-T_j}{b}\right) 
\end{equation*}

with:

* $s_j$, the height of the jump of the Kaplan-Meier estimator at $T_j$
* $K$, the infinite order kernel function. Here are the most common kernel functions:

    * `Biweight`: $f(x) = 0$ if $|x| \leq 1$ else $f(x)=\frac{15}{16}(1-x^2)^2$
    * `Cosine`:  $f(x) = 0$ if $|x| \leq 1$ else  $f(x)=\frac{\pi}{4}\cos( \frac{\pi x}{2} )$  
    * `Epanechnikov`: $f(x) = 0$ if $|x| \leq 1$ else $f(x) = 0.75 \cdot (1 - x^2 )$
    * `Normal`: $f(x) = \frac{\exp( -x^2/2)}{ \sqrt{ 2 \pi }}$
    * `Triweight`: $f(x) = 0$ if $|x| \leq 1$ else $f(x)=\frac{35}{32}(1-x^2)^3$
    * `Uniform`: $f(x) = 0$ if $|x|<1$ else $f(x) = 0.5$

* $b$, the kernel function bandwidth

---

## References
* [https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator)
* [Kaplan, E. L.; Meier, P. (1958). "Nonparametric estimation from incomplete observations". J. Amer. Statist. Assoc. 53 (282): 457–481. doi:10.2307/2281868. JSTOR 2281868.](https://web.stanford.edu/~lutian/coursepdf/KMpaper.pdf)
* [https://www.researchgate.net/publication/50940632_Understanding_survival_analysis_Kaplan-Meier_estimate](https://www.researchgate.net/publication/50940632_Understanding_survival_analysis_Kaplan-Meier_estimate)
* [survPresmooth: An R Package for PreSmooth Estimation in Survival Analysis](https://www.jstatsoft.org/article/view/v054i11)
* [Nonparametric density estimation from censored data](https://doi.org/10.1080/03610928408828780)
* [CDF and survival function estimation with infinite-order kernels](https://projecteuclid.org/download/pdfview_1/euclid.ejs/1261671304)
