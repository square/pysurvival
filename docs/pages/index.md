<!-- Index/Homepage -->
<style>
  h1, h2, h3 { color: #04A9F4; }
	pre { color: black !important; }
</style>

# Welcome to pysurvival.io

<center><img src="images/pysurvival_logo_black_blue.png" alt="PySurvival logo" title="PySurvival logo" width=100%, height=100%  /></center>


## What is PySurvival ?
PySurvival is an open source python package for Survival Analysis modeling - *the modeling concept used to analyze or predict when an event is likely to happen*. It is built upon the most commonly used machine learning packages such [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/) and [PyTorch](https://pytorch.org/).

PySurvival is compatible with Python 2.7-3.7.

---

## Content
PySurvival provides a very easy way to navigate between theoretical knowledge on Survival Analysis and detailed tutorials on how to conduct a full analysis, build and use a model. Indeed, the package contains:

* 10+ models ranging from the [Cox Proportional Hazard model](models/coxph.md), the [Neural Multi-Task Logistic Regression](models/neural_mtlr.md) to [Random Survival Forest](models/random_survival_forest.md)
* Summaries of the theory behind each model as well as API descriptions and examples.
* Tutorials displaying in great details how to perform exploratory data analysis, survival modeling, cross-validation and prediction, for [churn modeling](tutorials/churn.md) and [credit risk](tutorials/credit_risk.md) to name a few.
* Performance metrics to assess the models' abilities like [c-index](metrics/c_index.md) or [brier score](metrics/brier_score.md)
* Simple ways to [load and save models](miscellaneous/save_load.md)
* ... and more !

---

## Getting started

Because of its simple API, PySurvival has been built to provide a great user experience when it comes to modeling.
Here's a quick modeling example to get you started:

```python
# Loading the modules
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.datasets import Dataset
from pysurvival.utils.metrics import concordance_index

# Loading and splitting a simple example into train/test sets
X_train, T_train, E_train, X_test, T_test, E_test = \
	Dataset('simple_example').load_train_test()

# Building a CoxPH model
coxph_model = CoxPHModel()
coxph_model.fit(X=X_train, T=T_train, E=E_train, init_method='he_uniform', 
                l2_reg = 1e-4, lr = .4, tol = 1e-4)

# Building a MTLR model
mtlr = LinearMultiTaskModel()
mtlr.fit(X=X_train, T=T_train, E=E_train, init_method = 'glorot_uniform', 
         optimizer ='adam', lr = 8e-4)

# Checking the model performance
c_index1 = concordance_index(model=coxph_model, X=X_test, T=T_test, E=E_test )
print("CoxPH model c-index = {:.2f}".format(c_index1))

c_index2 = concordance_index(model=mtlr, X=X_test, T=T_test, E=E_test )
print("MTLR model c-index = {:.2f}".format(c_index2))
```

For additional models and performance metrics, checkout the documentation.

---

## Citation

If you use Pysurvival in your research and we would greatly appreciate if you could use the following:
<pre><code class="bash">@Misc{ pysurvival_cite,
  author = {Stephane Fotso and others},
  title = {{PySurvival}: Open source package for Survival Analysis modeling},
  year = {2019--},
  url = "https://www.pysurvival.io/",
  note = {[Online; accessed <today>]}
}</pre></code>
