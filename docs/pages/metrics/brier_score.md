<!-- # Metrics: C-index -->
<style>
  h1, h2, h3, h4 { color: #04A9F4; }
</style>

# Brier Score and Integrated Brier Score


## Brier Score


The **Brier score** is used to evaluate the accuracy of a predicted survival function at a given time $t$; it represents the *average squared distances between the observed survival status and the predicted survival probability* and is always a number between 0 and 1, with 0 being the best possible value. 

Given a dataset of $N$ samples,  $\forall i \in  [\![1, N ]\!],  \left(\vec{x}_i, \delta_i, T_i \right)$ is the format of a datapoint, and the predicted survival function is $ \hat{S}(t, \vec{x}_i), \forall t \in \mathbb{R^+}$:

In the absence of right censoring, the Brier score can be calculated such that:
		\begin{equation} 
			BS(t) = \frac{1}{N} \sum_{i = 1}^{N} (\mathbb{1}_{ T_i > t } - \hat{S}(t, \vec{x}_i))^2
		\end{equation}	
 
However, if the dataset contains samples that are right censored, then it is necessary to adjust the score by weighting the squared distances using the inverse probability of censoring weights method.
  Let $ \hat{G}(t) = P[C > t ]$ be the estimator of the conditional survival function of the censoring times calculated using the Kaplan-Meier method, where $C$ is the censoring time.
\begin{equation}  
BS(t) =  \frac{1}{N} \sum_{i = 1}^{N} \left( \frac{\left( 0 - \hat{S}(t, \vec{x}_i)\right)^2 \cdot \mathbb{1}_{T_i \leq t, \delta_i = 1}}{ \hat{G}(T_i^-)} + \frac{ \left( 1 - \hat{S}(t, \vec{x}_i)\right)^2 \cdot \mathbb{1}_{T_i > t}}{ \hat{G}(t)} \right)
\end{equation}	

In terms of benchmarks, a useful model will have a Brier score below $0.25$. Indeed,  it is easy to see that if $\forall i \in [\![1, N]\!], \hat{S}(t, \vec{x}_i) = 0.5$, then $BS(t) = 0.25$.


---

### Location

The function can be found at `pysurvival.utils.metrics.brier_score`.

---

### API

!!! abstract "`brier_score`  - Brier score computations"
	```
		brier_score(model, X, T, E, t_max=None)
	```

	**Parameters:**

    * `model` : **Pysurvival object** --
        Pysurvival model

    * `X` : **array-like** --
        input samples; where the rows correspond to an individual sample and the columns represent the features *(shape=[n_samples, n_features])*.

    * `T` : **array-like** -- 
        target values describing the time when the event of interest or censoring
        occurred.

    * `E` : **array-like** --
        values that indicate if the event of interest occurred i.e.: E[i]=1
        corresponds to an event, and E[i] = 0 means censoring, for all i.

	* `t_max`: **float** *( default=None )*
		Maximal time for estimating the prediction error curves. 
		If missing the largest value of the response variable is used.

	**Returns:**

	* `times`: array-like.
		A vector of timepoints. At each timepoint the brier score is estimated

	* `brier_scores`: array-like.
		A vector of brier scores

---

## Integrated Brier Score


The **Integrated Brier Score** (IBS) provides an overall calculation of the model performance at all available times.
\begin{equation}  
	\text{IBS}(t_{\text{max}}) =  \frac{1}{t_{\text{max}}}   \int_{0}^{t_{\text{max}}}  BS(t) dt
\end{equation}	

---

### Location
The function can be found at `pysurvival.utils.metrics.integrated_brier_score` to output the values and `pysurvival.utils.display.integrated_brier_score` to display the predictive error curve.

---

### API

!!! abstract "`integrated_brier_score`  - Integrated Brier score computations"
	```
	integrated_brier_score(model, X, T, E, t_max=None, figure_size=(20, 10))
	```

	**Parameters:**

    * `model` : **Pysurvival object** --
        Pysurvival model

    * `X` : **array-like** --
        input samples; where the rows correspond to an individual sample and the columns represent the features *(shape=[n_samples, n_features])*.

    * `T` : **array-like** -- 
        target values describing the time when the event of interest or censoring
        occurred.

    * `E` : **array-like** --
        values that indicate if the event of interest occurred i.e.: E[i]=1
        corresponds to an event, and E[i] = 0 means censoring, for all i.

	* `t_max`: **float** *( default=None )*
		Maximal time for estimating the prediction error curves. 
		If missing the largest value of the response time variable is used.

    * `figure_size`: **tuple of double** *( default=(20, 10) )*
        width, height in inches representing the size of the chart of 
        the survival function.
        Option available if the function is being called from `pysurvival.utils.display`

	**Returns:**

	* `ibs`: **double**
		The integrated brier score
