"""Custom LabelEncoder for keep an order of labels.
"""

# %%
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import LabelEncoder


def _encode_numpy_with_int(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        if encode:
            # uniques, encoded = np.unique(values, return_inverse=True)
            uniques = pd.unique(values)
            _, encoded = np.unique(values, return_inverse=True)
            return uniques, encoded
        else:
            # # unique sorts
            # return np.unique(values)
            return pd.unique(values)
    if encode:
        diff = skl.preprocessing.label._encode_check_unknown(values, uniques)
        if diff:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(diff))
        # encoded = np.searchsorted(uniques, values)
        table = {val: i for i, val in enumerate(uniques)}
        # encoded = np.fromiter((table[v] for v in values), dtype='int64')
        encoded = np.array([table[v] for v in values])
        return uniques, encoded
    else:
        return uniques


def _custom_encode(values, uniques=None, encode=False):
    """Helper function to factorize (find uniques) and encode values.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.
    Parameters
    ----------
    values : array
        Values to factorize or encode.
    uniques : array, optional
        If passed, uniques are not determined from passed values (this
        can be because the user specified categories, or because they
        already have been determined in fit).
    encode : bool, default False
        If True, also encode the values into integer codes based on `uniques`.
    Returns
    -------
    uniques
        If ``encode=False``. The unique values are sorted if the `uniques`
        parameter was None (and thus inferred from the data).
    (uniques, encoded)
        If ``encode=True``.
    """
    if values.dtype == object:
        try:
            res = skl.preprocessing.label._encode_python(values, uniques, encode)
        except TypeError:
            raise TypeError("argument must be a string or number")
        return res
    elif values.dtype == np.integer:
        return _encode_numpy_with_int(values, uniques, encode)
    else:
        return skl.preprocessing.label._encode_numpy(values, uniques, encode)


class IntLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()

    def fit(self, y):
        """Fit label encoder
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """
        y = skl.utils.column_or_1d(y, warn=True)
        # self.classes_ = _encode(y)
        # self.classes_ = pd.Series(y).unique()
        self.classes_ = _custom_encode(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        y = skl.utils.column_or_1d(y, warn=True)
        # self.classes_, y = _encode(y, encode=True)
        # self.classes_ = pd.Series(y).unique()
        # y = self.transform(y)
        self.classes_, y = _custom_encode(y, encode=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        skl.utils.validation.check_is_fitted(self, 'classes_')
        y = skl.utils.column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if skl.utils.validation._num_samples(y) == 0:
            return np.array([])

        # _, y = _encode(y, uniques=self.classes_, encode=True)
        _, y = _custom_encode(y, uniques=self.classes_, encode=True)
        return y


# %% Test --------------------------------------------------------------------

le = IntLabelEncoder().fit([2, 1, 6])
le.classes_
le.transform([6, 6, 2, 1])
le.inverse_transform([0, 2, 0, 1])
