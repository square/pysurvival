<!-- # Survival Forest models-->
<style>
  h1, h2, h3 { color: #04A9F4; }
</style>


# Survival Forest models

The Ensemble models that use decision trees as its base learners can be extended to take into account censored datasets. These types of models can be regrouped under the name **Survival Forest models**. PySurvival contains 3 types of Survival Forest models:

* Random Survival Forest model ([`RandomSurvivalForestModel`](random_survival_forest.md))
* Extremely Randomized (Extra) Survival Trees model ([`ExtraSurvivalTreesModel`](extra_survival_trees.md))
* Conditional Survival Forest model ([`ConditionalSurvivalForestModel`](conditional_survival_forest.md))

These models have been adapted to python from the package [ranger](https://arxiv.org/pdf/1508.04409.pdf), which is a fast implementation of random forests in C++.

---

## General algorithm
[Ishwaran et al.](https://arxiv.org/pdf/0811.1645.pdf) provides a general framework that can be used to describe the underlying algorithm that powers  the Survival Forest models:

1. Draw $B$ random samples of the same size from the original dataset with replacement. The samples that are not
drawn are said to be out-of-bag (OOB).

2. Grow a survival tree on each of the $b = 1, ..., B$ samples. 

	**a.** At each node, select a random subset of predictor variables and find the best
	predictor and splitting value that provide two subsets (the daughter nodes) 
	which maximizes the difference in the objective function.

	**b.** Repeat **a.** recursively on each daughter node until a stopping criterion is met.

3. Calculate a cumulative hazard function (CHF) for each tree and average over all CHFs
for the B trees to obtain the ensemble CHF.

4. Compute the prediction error for the ensemble CHF using only the OOB data.

All the Survival Forest models in PySurvival use this framework as the basis of the model fitting algorithm. The objective function is the main element that can differentiate then from one another.

---

## [Random Survival Forest model](random_survival_forest.md)

At each node, we choose a predictor **$x$** from a subset of randomly selected predictor variables and a split value **$c$**.
**$c$** is one of the unique values of **$x$**

We assign each individual sample $i$ to either the right node, if $x_i \leq c$ or left daughter node if $x_i > c$.
Then we calculate the value of the log rank test such that:

\begin{equation}
L(x, c) = \frac{ \sum^{N}_{i=1} \left(  d_{i, 1} - Y_{i,1} \frac{d_i}{Y_i} \right)  }
			   { \sqrt{  \sum^{N}_{i=1}  \frac{Y_{i,1}}{Y_i} \left( 1 - \frac{Y_{i,1}}{Y_i} \right) \left( \frac{Y_i-d_i}{Y_i-1} \right) d_i  } }
\end{equation}

with:

* $j$: Daughter node, $j \in \{1, 2\}$
* $d_{i,j}$ : Number of events at time $t_i$ in daughter node $j$.
* $Y_{i,j}$ : Number of units that experienced an event or are at risk at time $t_i$ in daughter node $j$.
* $d_i$ : Number of events at time $t_i$, so $d_i=\sum_j d_{i,j}$
* $Y_i$ : Number of units that experienced an event or at risk at time $t_i$, so $Y_i=\sum_j Y_{i,j}$

We loop through every **$x$** and **$c$** until we find $x^{*}$ and $c^{*}$ that satisfy $|L(x^{*}, c^{*})| \geq |L(x, c)|$ for
every $x$ and $c$.

---

## [Extremely Randomized (Extra) Survival Trees model](extra_survival_trees.md)

Extra Survival Trees models use the same objective function as the Random Survival Forest models. But for each predictor **$x$**, instead of using the unique values of **$x$** to find the best split value $c^{*}$, we use $N_{splits}$ values drawn from a uniform distribution over the interval $\left[\min(x), \max(x)\right]$.

---

## [Conditional Survival Forest model](conditional_survival_forest.md)

Conditional Survival Forest models are constructed in a way that is a bit different from Random Survival Forest models:

1. The objective function is given by testing the null hypothesis that there is independence between
the response and the predictor. To do so, for each predictor variable $x$, compute the logrank score test statistic and its associated p-value:
	* Let's consider $n$ observations $(T_1, \delta_1), ... , (T_n, \delta_n)$. We will assume the predictor $x$ has been ordered so that $x_1 \leq x_2 \leq ... \leq x_n$. With $\gamma_j = \sum^n_{i=1} 1_{T_i \leq T_j}$, we compute the logrank scores $a_1, ..., a_n$ such that :

		\begin{equation}
		a_i = \delta_i - \sum^{\gamma_i}_{j=1} \frac{\delta_j}{(n-\gamma_j + 1)}
		\end{equation}

	* For a predictor $x$ and split value $c$, and within the right node ($x_i \leq c$), we can now calculate :
		- the sum of all scores $S_{n, c} = \sum_{i=1}^n 1_{x_i \leq c} \cdot a_i$
		- its expectation $\text{E}\left[ S_{n, c} \right] = m_c \cdot \overline{a}$  with  $m_c=\sum_{i=1}^n 1_{x_i \leq c}$ and $\overline{a}=\frac{1}{n}\sum_{i=1}^n a_i$
		- its variance $\text{Var}\left[ S_{n, c} \right] = \frac{m_c n_c}{n(n-1)} \sum_{i=1}^n \left( a_i - \overline{a} \right)^2$  with $n_c = n-m_c$

	* We can obtain the score test statistic $T_{n,c} = \frac{ S_{n, c} - \text{E}\left[ S_{n, c} \right] }{ \sqrt{\text{Var}\left[ S_{n, c} \right] } }$ and look for $c^{*}$ such that $|T_{n, c^{*}}| \geq |T_{n, c}|$.

	* Finally, we compute the p-value associated with $T_{n, c^{*}}$.

2. At each node, only for the predictors whose associated p-value is smaller than a specified value $\alpha$, the predictor with the smallest p-value is selected as splitting candidate. However, if no predictor can be used then no split is performed.


## References

* [Ishwaran H, Kogalur U, Blackstone E, Lauer M. Random survival forests. The Annals of Applied
Statistics. 2008; 2(3):841–860.](https://arxiv.org/pdf/0811.1645.pdf)
* [ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R](https://arxiv.org/pdf/1508.04409.pdf)
* [Weathers, Brandon and Cutler, Richard Dr., "Comparison of Survival Curves Between Cox Proportional Hazards, Random Forests, and Conditional Inference Forests in Survival Analysis" (2017). All Graduate Plan B and other Reports. 927.](https://digitalcommons.usu.edu/gradreports/927)
* [Wright, Marvin N., Theresa Dankowski and Andreas Ziegler. "Random forests for survival analysis using maximally selected rank statistics."" Statistics in medicine 36 8 (2017): 1272-1284.](https://arxiv.org/pdf/1605.03391.pdf)
* [Geurts, Pierre & Ernst, Damien & Wehenkel, Louis. (2006). Extremely Randomized Trees. Machine Learning. 63. 3-42. 10.1007/s10994-006-6226-1.](https://link.springer.com/content/pdf/10.1007/s10994-006-6226-1.pdf)
