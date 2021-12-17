# Python codes for Deep Learning

You can import this directory as a module, or run notebooks in this directory.

0. Generate data using codes in the `HIV_simulator` directory. 

1. Train models using `MakeAndTrain_CLI.py`. Modify to provide desired hyperparameters. Note where the models are saved.

2. Import the trained model. 

```python
from Structures import MultipleModel

model=MultipleModel()
model.import_from_directory("./path/to/models/")
```

3. Import data for analysis using `RealData.RealData`. See methods `predict_by_year`, `show`, `lineplot`, and `predict`.
```python
import RealData as RD
```


4. Permutation analysis can be run on a specific set the function `permutation_test` from the `Permutation` module.
```python
from Permutation import permutation_test
```

5. Time series analyses can be performed. Evolutionary distance calculations and subsetting are not handled here. Generate a directory of `.mat` files. File nomenclature is described in the `TimeSeries` documentation. 

```python
from TimeSeries import TimeSeries
```



