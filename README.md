# dapqa-fast-mf

Fast implementations of SVD and SVD++ matrix factorization algorithms.  
Based on [Numba](https://numba.pydata.org/).

## Benchmarking

To run benchmarks, execute

```shell
python -m pytest benchmark --benchmark-autosave
```

in a virtual environment console in the root directory.

To view or compare existing benchmark results, run 

```shell
py.test-benchmark compare .benchmarks/Windows-CPython-3.9-64bit/saved-file-name.json
```

