import numpy as np
from numba import jit, types

from sklearn.base import BaseEstimator, RegressorMixin


@jit(
    types.Tuple((
            types.Array(types.float64, 2, 'C'),
            types.Array(types.float64, 2, 'C'),
            types.Array(types.float64, 1, 'C'),
            types.Array(types.float64, 1, 'C'),
            types.float64
    ))(
        types.Array(types.int64, 2, 'C'),
        types.Array(types.float64, 1, 'C'),
        types.Array(types.int64, 1, 'C'),
        types.Array(types.int64, 1, 'C'),
        types.int64,
        types.int64,
        types.float64,
        types.float64
    ),
    locals={
        'mean_rating': types.float64,
        'user_factors': types.Array(types.float64, 2, 'C'),
        'item_factors': types.Array(types.float64, 2, 'C'),
        'user_biases': types.Array(types.float64, 1, 'C'),
        'item_biases': types.Array(types.float64, 1, 'C'),
        'row': types.Array(types.int64, 1, 'C'),
        'original_row': types.Array(types.int64, 1, 'C'),
        'r_ui': types.int64,
        'u_idx': types.int64,
        'i_idx': types.int64,
        'b_u': types.float64,
        'b_i': types.float64,
        'p_u': types.Array(types.float64, 1, 'C'),
        'q_i': types.Array(types.float64, 1, 'C'),
        'r_ui_pred': types.float64,
        'e_ui': types.float64,
    },
    nopython=True,
    fastmath=True,
    cache=True,
)
def _fit_svd(
        X_train_widx,
        y_train,
        user_ids_uq,
        item_ids_uq,
        n_factors,
        n_epochs,
        learning_rate,
        reg
):
    mean_rating = np.mean(y_train)

    user_factors = np.random.normal(size=(len(user_ids_uq), n_factors), loc=0, scale=.1)
    item_factors = np.random.normal(size=(len(item_ids_uq), n_factors), loc=0, scale=.1)
    user_biases = np.zeros(len(user_ids_uq))
    item_biases = np.zeros(len(item_ids_uq))

    b_u = user_biases[0]
    b_i = item_biases[0]
    p_u = user_factors[0]
    q_i = item_factors[0]
    prev_u_idx = 0
    prev_i_idx = 0
    for epoch_number in range(n_epochs):
        for i in range(len(X_train_widx)):
            r_ui = y_train[i]
            row = X_train_widx[i]
            u_idx, i_idx = row[0], row[1]

            # Reading/Writing variables

            if prev_u_idx != u_idx:
                user_biases[prev_u_idx] = b_u
                user_factors[prev_u_idx] = p_u
                b_u = user_biases[u_idx]
                p_u = user_factors[u_idx]

                prev_u_idx = u_idx

            if prev_i_idx != i_idx:
                item_biases[prev_i_idx] = b_i
                item_factors[prev_i_idx] = q_i
                b_i = item_biases[i_idx]
                q_i = item_factors[i_idx]

                prev_i_idx = i_idx

            # Calculating the prediction and its error
            r_ui_pred = mean_rating + b_u + b_i + np.dot(p_u, q_i)
            e_ui = r_ui - r_ui_pred

            # Updating biases
            b_u += learning_rate * (e_ui - reg * b_u)
            b_i += learning_rate * (e_ui - reg * b_i)

            # Updating factors
            for j in range(n_factors):
                p_u_j = p_u[j]
                q_i_j = q_i[j]
                p_u[j] = p_u_j + learning_rate * (e_ui * q_i_j - reg * p_u_j)
                q_i[j] = q_i_j + learning_rate * (e_ui * p_u_j - reg * q_i_j)

    # Biases and factors are not updated at the last iteration, so here the manual last update
    user_biases[prev_u_idx] = b_u
    user_factors[prev_u_idx] = p_u
    item_biases[prev_i_idx] = b_i
    item_factors[prev_i_idx] = q_i

    return user_factors, item_factors, user_biases, item_biases, mean_rating


@jit(
    types.Array(types.float64, 1, 'C')(
        types.Array(types.int64, 2, 'C'),
        types.Array(types.int64, 2, 'C'),
        types.Array(types.int64, 1, 'C'),
        types.Array(types.int64, 1, 'C'),
        types.Array(types.float64, 2, 'C'),
        types.Array(types.float64, 2, 'C'),
        types.Array(types.float64, 1, 'C'),
        types.Array(types.float64, 1, 'C'),
        types.float64
    ),
    nopython=True,
    fastmath=True,
    cache=True,
)
def _predict_svd(
        X_test_widx,
        X_test,
        user_ids_uq,
        item_ids_uq,
        user_factors,
        item_factors,
        user_biases,
        item_biases,
        mean_rating
):
    y_pred = np.full((len(X_test_widx)), mean_rating)

    for i in range(len(X_test_widx)):
        row = X_test_widx[i]
        original_row = X_test[i]

        u_idx, i_idx = row[0], row[1]
        if u_idx < 0 or u_idx >= len(user_ids_uq) or user_ids_uq[u_idx] != original_row[0] \
                or i_idx < 0 or i_idx >= len(item_ids_uq) or item_ids_uq[i_idx] != original_row[1]:
            continue

        b_u = user_biases[u_idx]
        b_i = item_biases[i_idx]
        p_u = user_factors[u_idx, :]
        q_i = item_factors[i_idx, :]

        y_pred[i] = mean_rating + b_u + b_i + np.dot(p_u, q_i)

    return y_pred


class SVD(BaseEstimator, RegressorMixin):
    def __init__(self,
                 user_ids_uq=None,
                 item_ids_uq=None,
                 n_factors=100,
                 reg=0.02,
                 learning_rate=0.005,
                 n_epochs=20
                 ) -> None:
        super().__init__()

        self.user_ids_uq = user_ids_uq
        self.item_ids_uq = item_ids_uq
        self.n_factors = n_factors
        self.reg = reg
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.mean_rating = 0

    def fit(self, X, y):
        if self.user_ids_uq is None:
            self.user_ids_uq = np.unique(X[:, 0])

        if self.item_ids_uq is None:
            self.item_ids_uq = np.unique(X[:, 1])

        X_widx = np.c_[
            np.searchsorted(self.user_ids_uq, X[:, 0]).T,
            np.searchsorted(self.item_ids_uq, X[:, 1]).T
        ]

        self.user_factors, self.item_factors, self.user_biases, self.item_biases, self.mean_rating = _fit_svd(
            X_train_widx=X_widx.astype('int64'),
            y_train=y.astype('float64'),
            user_ids_uq=self.user_ids_uq.astype('int64'),
            item_ids_uq=self.item_ids_uq.astype('int64'),
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            reg=self.reg
        )

    def predict(self, X):
        X_widx = np.c_[
            np.searchsorted(self.user_ids_uq, X[:, 0]).T,
            np.searchsorted(self.item_ids_uq, X[:, 1]).T
        ]

        return _predict_svd(
            X_test_widx=X_widx.astype('int64'),
            X_test=X.astype('int64'),
            user_ids_uq=self.user_ids_uq.astype('int64'),
            item_ids_uq=self.item_ids_uq.astype('int64'),
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            mean_rating=self.mean_rating
        )
