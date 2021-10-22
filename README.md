# dapqa-fast-mf

Fast implementations of SVD and SVD++ matrix factorization algorithms.  
Powered by [Numba](https://numba.pydata.org/).

## Usage

Models available are `SVD`, `SVDpp`.  
All the parameters are set via constructor.

Input data contain two columns -- `user_id` and `item_id`.  
Output data contain ratings.

**IMPORTANT**  
Data must be sorted by `user_id`.  
To sort data, there is a helper function `dfmf.util.sort_by_user`.

Example usage:

```python
import numpy as np
from sklearn.model_selection import train_test_split

from dfmf.model import SVD
from dfmf.util import sort_by_user

data = np.genfromtxt('my-fancy-dataset.csv', delimiter=',')
X, y = data[:, 0:2], data[:, 2]

X_train, X_test, y_train, y_test = sort_by_user(
    *train_test_split(X, y, test_size=0.2)
)

model = SVD()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

```

## Benchmarking

To run benchmarks, execute

```shell
python -m pytest benchmark --benchmark-autosave
```

in a virtual environment console in the root directory.  
As in raw pytest, keywords can be used to filter benchmarks. For example, only ml100k benchmarks can be run using

```shell
python -m pytest benchmark --benchmark-autosave -k ml100k
```

To view or compare existing benchmark results, run

```shell
py.test-benchmark compare .benchmarks/Windows-CPython-3.9-64bit/saved-file-name.json
```

