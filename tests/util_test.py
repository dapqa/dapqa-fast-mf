import pytest
import numpy as np

from dfmf.util import sort_by_user


def test_sort_all_data():
    data = np.array([
        [2, 2, 4],
        [1, 1, 3],
        [3, 2, 5]
    ])
    expected = np.array([
        [1, 1, 3],
        [2, 2, 4],
        [3, 2, 5]
    ])

    sorted = sort_by_user(data)
    assert (sorted == expected).all()


def test_sort_inputs_and_outputs_separately():
    X1 = np.array([
        [2, 2],
        [1, 1],
        [3, 2]
    ])
    y1 = np.array([4, 3, 5])

    expected_X1 = np.array([
        [1, 1],
        [2, 2],
        [3, 2]
    ])
    expected_y1 = np.array([3, 4, 5])

    # ---

    X2 = np.array([
        [5, 2],
        [5, 3],
        [4, 3]
    ])
    y2 = np.array([4, 5, 3])

    expected_X2 = np.array([
        [4, 3],
        [5, 2],
        [5, 3]
    ])
    expected_y2 = np.array([3, 4, 5])

    # ---

    X3 = np.array([
        [2, 2],
        [1, 2]
    ])
    y3 = np.array([5, 4])

    expected_X3 = np.array([
        [1, 2],
        [2, 2]
    ])
    expected_y3 = np.array([4, 5])

    # ---

    sorted_X1, sorted_y1, sorted_X2, sorted_y2, sorted_X3, sorted_y3 = sort_by_user(X1, y1, X2, y2, X3, y3)

    assert (sorted_X1 == expected_X1).all()
    assert (sorted_y1 == expected_y1).all()
    assert (sorted_X2 == expected_X2).all()
    assert (sorted_y2 == expected_y2).all()
    assert (sorted_X3 == expected_X3).all()
    assert (sorted_y3 == expected_y3).all()

    assert (sorted_X1 != X1).any()
    assert (sorted_y1 != y1).any()

    # ---

    sorted_X1, sorted_X2, sorted_X3, sorted_y1, sorted_y2, sorted_y3 = sort_by_user(X1, X2, X3, y1, y2, y3)

    assert (sorted_X1 == expected_X1).all()
    assert (sorted_y1 == expected_y1).all()
    assert (sorted_X2 == expected_X2).all()
    assert (sorted_y2 == expected_y2).all()
    assert (sorted_X3 == expected_X3).all()
    assert (sorted_y3 == expected_y3).all()


def test_sort_with_invalid_arrays():
    invalid_arr = np.array([
        [
            [1, 1, 1],
            [2, 2, 2]
        ]
    ])

    assert len(invalid_arr.shape) == 3

    with pytest.raises(ValueError):
        _ = sort_by_user(invalid_arr)
