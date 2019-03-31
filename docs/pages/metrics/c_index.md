<!-- # Metrics: C-index -->
<style>
  h1, h2, h3, h4 { color: #04A9F4; }
</style>

# C-index


## Introduction

The **concordance index** or **C-index** is *a generalization of the area under the ROC curve (AUC)* that can take into account censored data. It represents the global assessment of the model discrimination power: this is the model’s ability to correctly provide a reliable ranking of the survival times based on the individual risk scores. It can be computed with the [following formula](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3079915/pdf/nihms255748.pdf):
\begin{equation} 
	\text{C-index} = \frac{ \sum_{i, j} \mathbb{1}_{T_j < T_i} \cdot \mathbb{1}_{\eta_j > \eta_i} \cdot \delta_j }{\sum_{i, j} \mathbb{1}_{T_j < T_i}\cdot \delta_j }
\end{equation}	 

with:

* $\eta_i$, the risk score of a unit $i$
* $\mathbb{1}_{ T_j < T_i } = 1$ if $T_j < T_i$ else $0$ 
* $\mathbb{1}_{ \eta_j > \eta_i } =  1$ if  $\eta_j > \eta_i$ else $0$ 


Similarly to the AUC, $\text{C-index}= 1$ corresponds to the best model prediction, and  $\text{C-index} = 0.5$ represents a random prediction. 

---

## Location
The function can be found at `pysurvival.utils.metrics.concordance_index`.

---

## API

!!! abstract "`concordance_index`  - Concordance Index computations"
	``` 
		concordance_index(model, X, T, E, include_ties = True, additional_results=False)
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
	
	* `include_ties`: **bool** *(default=True)* -- 
		Specifies whether ties in risk score are included in calculations

	* `additional_results`: **bool** *(default=False)* -- 
		Specifies whether only the c-index should be returned (False)
		or if a dict of values should returned. In that case, the values are:

		- c_index
		- nb_pairs
		- nb_concordant_pairs

	**Returns:**

	* `c_index`: **float** or **dict** -- Result of the function

		- if `additional_results = False` then c_index is **float**.
		- if `additional_results = True` then c_index is **dict**, such that `c_index = {'c_index': ., 'nb_pairs': ., 'nb_concordant_pairs': .}`


## References:
* [Uno, Hajime et al. “On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data” Statistics in medicine vol. 30,10 (2011): 1105-17.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3079915/pdf/nihms255748.pdf)