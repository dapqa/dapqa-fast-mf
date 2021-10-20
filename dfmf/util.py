import numpy as np


def _is_input_array(array):
    if len(array.shape) > 2:
        raise ValueError(f'Expected 1-dimensional or 2-dimensional arrays, got {len(array.shape)}-dimensional')

    return len(array.shape) == 2 and array.shape[1] > 1


def sort_by_user(*arrays):
    res = [None for _ in range(len(arrays))]

    output_arr_idx = -1
    for input_arr_idx in range(len(arrays)):
        X = arrays[input_arr_idx]
        if not _is_input_array(X):
            continue

        sort_perm = np.argsort(X[:, 0])
        res[input_arr_idx] = X[sort_perm, :]

        for output_arr_idx in range(max(output_arr_idx, input_arr_idx) + 1, len(arrays)):
            y = arrays[output_arr_idx]
            if _is_input_array(y):
                continue

            res[output_arr_idx] = y[sort_perm]
            break

    return res
