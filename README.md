# PySurvival

<center><img src="docs/pages/images/pysurvival_logo_black_blue.png" alt="pysurvival_logo" title="pysurvival_logo" width="50%", height="50%" /></center>

## What is Pysurvival ?
PySurvival is an open source python package for Survival Analysis modeling - *the modeling concept used to analyze or predict when an event is likely to happen*. It is built upon the most commonly used machine learning packages such [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/) and [PyTorch](https://pytorch.org/).

PySurvival is compatible with Python 2.7-3.7.

Check out the documentation [here](https://www.pysurvival.io)

---

## Content
PySurvival provides a very easy way to navigate between theoretical knowledge on Survival Analysis and detailed tutorials on how to conduct a full analysis, build and use a model. Indeed, the package contains:

* 10+ models ranging from the [Cox Proportional Hazard model](https://www.pysurvival.io/models/coxph.html), the [Neural Multi-Task Logistic Regression](https://www.pysurvival.io/models/neural_mtlr.html) to [Random Survival Forest](https://www.pysurvival.io/models/random_survival_forest.html)
* Summaries of the theory behind each model as well as API descriptions and examples.
* Tutorials displaying in great details how to perform exploratory data analysis, survival modeling, cross-validation and prediction, for [churn modeling](https://www.pysurvival.io/tutorials/churn.html) and [credit risk](https://www.pysurvival.io/tutorials/credit_risk.html) to name a few.
* Performance metrics to assess the models' abilities like [c-index](https://www.pysurvival.io/metrics/c_index.html) or [brier score](https://www.pysurvival.io/metrics/brier_score.html)
* Simple ways to [load and save models](https://www.pysurvival.io/miscellaneous/save_load.html)
* ... and more !

---

## Installation

If you have already installed a working version of gcc, the easiest way to install Pysurvival is using pip
```
pip install pysurvival
```
The full description of the installation steps can be found [here](https://www.pysurvival.io/installation.html).

---

## Get Started

Because of its simple API, Pysurvival has been built to provide to best user experience when it comes to modeling.
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

---

## Citation and License

### Citation
If you use Pysurvival in your research and we would greatly appreciate if you could use the following:

```
@Misc{ pysurvival_cite,
  author =    {Stephane Fotso and others},
  title =     {PySurvival: Open source package for Survival Analysis modeling},
  year =      {2019--},
  url = "https://www.pysurvival.io/"
}
```

### License

Copyright 2019 Square Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
