<!-- Saving and Loading -->
<style>
  h1, h2, h3, h4 { color: #04A9F4; }
</style>

# Saving and Loading a model

Saving or Loading a model is very straighforward in pySurvival.

## Saving 
To save a model, use the function `save_model` and provide the full path of the future location of the file as the argument; the model is then compressed into a `.zip` file. The function is located at `pysurvival.utils.save_model`.

### API

!!! abstract " `save_model` - Save and compress the model and its parameters into a .zip file "

    ```
    save_model(path_file)
    ```

    **Parameters:**

    * `model` : **Pysurvival object** -- 
        Pysurvival model

    * `path_file`, **str** -- 
        full address of the file where the model will be saved 


### Example
```python
# Importing modules
from pysurvival.models.svm import KernelSVMModel
from pysurvival.datasets import Dataset

# Loading and splitting a simple example into train/test sets
X_train, T_train, E_train, \
    X_test, T_test, E_test = Dataset('simple_example').load_train_test()

# Let's assume we want to build the following SVM model
svm_model = KernelSVMModel('gaussian')
svm_model.fit(X_train, T_train, E_train, init_method='glorot_uniform',
              l2_reg = 1e-5, lr = 0.5)

# Let's now save our model
from pysurvival.utils import save_model
save_model(svm_model, '/Users/xxx/Desktop/svm_model_2018_08_26.zip')
```

---

## Loading 
To load a model, use the function `load_model` and provide the full path of the location of the file as the argument.  The function is located at `pysurvival.utils.load_model`.


### API

!!! abstract " `load_model` - Load the model and its parameters from a .zip file "

    ```
    load_model(path_file)
    ```

    **Parameters:**

    * `path_file` : **str** -- 
        full address of the file where the model will be loaded from 


### Example
```python
# Let's assume we have built and saved a SVM model at the following location
# /Users/xxx/Desktop/svm_model_2018_08_26.zip

from pysurvival.utils import load_model
svm_model = load_model('/Users/xxx/Desktop/svm_model_2018_08_26.zip')
```