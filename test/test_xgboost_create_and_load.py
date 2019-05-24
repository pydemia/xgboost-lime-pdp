
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import sklearn as skl
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from custom_labelencoder import IntLabelEncoder
from sklearn.metrics import accuracy_score

import xgboost as xgb
# from xgboost.compat import XGBLabelEncoder
# => is IDENTICAL to `sklearn.preprocessing.LabelEncoder`

import lime
from lime import lime_tabular

import pdpbox
from pdpbox import pdp, info_plots


for _pkg in [np, pd, skl, xgb, mpl, pdpbox]:
    print(f'{_pkg.__name__:<7} = {_pkg.__version__}')

font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'dejavu' in path.lower().split('/')[-1]
}

plt.rcParams['font.family'] = sorted(font_dict.keys(), key=len)[0]
os.chdir('../git/xgboost-lime-pdp')
fpath = '.'


# %% Classes & Functions -----------------------------------------------------

def as_int(string):
    return np.fromstring(string, dtype=np.int64, sep=',')[0]


def as_int_str(string):
    return np.fromstring(string, dtype=np.int64, sep=',').astype(np.str)[0]


def as_str(string):
    return np.fromstring(string, dtype=np.str, sep=',')[0]


class DataFrameImputer(TransformerMixin):
    """Fill missing values.

    Columns of dtype object are imputed with the most frequent value
    in column.
    Columns of other types are imputed with mean of column.

    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):

        self.fill = pd.Series(
            [
                X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O')
                else X[c].mean()
                for c in X
            ],
            index=X.columns,
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def preprocess_by_column(dataframe):
    """Column Typing & Categorization.

    If a column has `float`, it will be min-max scaled.
    If a column has `str`, it will be categorized & substitute by int, which is its code number.
    If a column has `int`, do nothing.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Data to preprocess.

    Return
    ------
    dataframe: pandas.DataFrame
        Preprocessed Data.

    col_dict: dict
        A dictionary of column lists by its dtype.

    float_scaler: sklearn.preprocessing.MinMaxScaler
        A MinMaxScaler instance of `float` type columns.

    """
    grouped = dataframe.columns.to_series().groupby(dataframe.dtypes).groups
    col_types = {
        dtype.name: colname_list.tolist()
        for dtype, colname_list in grouped.items()
    }
    float_columns = col_types.get('float64', [])
    int_columns = col_types.get('int64', [])
    str_columns = col_types.get('object', [])
    cat_columns = int_columns + str_columns

    if bool(cat_columns):
        dataframe[cat_columns] = dataframe[cat_columns].astype('category')
        category_codes_df = (
            dataframe[cat_columns].apply(lambda x: x.cat.codes)
        )
        category_names_df = (
            dataframe[cat_columns].apply(lambda x: x.cat.categorical)
        )
        category_dict = (
            dataframe
            [cat_columns]
            .agg(lambda x: [x.cat.categories.tolist()])
            .to_dict('records')
        )[0]
        dataframe[cat_columns] = category_codes_df

    else:
        category_dict = {}

    if float_columns is not None:
        float_scaler = MinMaxScaler(feature_range=(0, 1))
        dataframe[float_columns] = float_scaler.fit_transform(
            dataframe[float_columns]
        )

    col_dict = {
        'float': float_columns,
        'category': cat_columns,
    }

    return dataframe, col_dict, category_dict, float_scaler


# %% LOAD DATA: CASE in {1, 2} -----------------------------------------------

CASE = 2
COLNAMES_TO_FLIKE = True

if int(CASE) == 1:
    DUMP_PATH = f'{fpath}/data/nativeBoost2'
    train_filename = f'{fpath}/data/train_63qYitG.csv'
    test_filename = f'{fpath}/data/test_XaoFywY.csv'

elif int(CASE) == 2:
    DUMP_PATH = f'{fpath}/data/nativeBoost6features'
    train_filename = f'{fpath}/data/train_new.csv'
    test_filename = f'{fpath}/data/test_new.csv'

else:
    raise ValueError('`CASE` Should be int `1` or int `2`.')


# DATA LOADING ---------------------------------------------------------------

if int(CASE) == 1:

    int_type_colnames = ['Surge_Pricing_Type', 'Var3']
    col_converter = {colname: as_int_str for colname in int_type_colnames}

    train_df = pd.read_csv(
        train_filename, header=0, converters=col_converter,
    )
    test_df = pd.read_csv(
        test_filename, header=0, converters=col_converter,
    )

    data = pd.concat(
        [train_df, test_df],
        axis=0,
        ignore_index=True,
    )
    data = DataFrameImputer().fit_transform(data)
    data = data.drop(
        [
            'Trip_ID',
            'Cancellation_Last_1Month',
            'Confidence_Life_Style_Index',
            'Gender',
            'Life_Style_Index',
            'Var1',
            'Var2',
        ],
        axis=1,
    )

    y_cols = target_names = ['Surge_Pricing_Type']


elif int(CASE) == 2:
    int_type_colnames = ['c1', 'c2', 'c6', 'r1']
    col_converter = {colname: as_int_str for colname in int_type_colnames}

    X = train_df = pd.read_csv(
        train_filename, header=0, converters=col_converter,
    )
    Y = test_df = pd.read_csv(
        test_filename, header=0, converters=col_converter,
    )
    data = pd.concat([X, Y], axis=1)

    y_cols = target_names = ['r1']


x_cols = feature_names = data.columns.drop(target_names).tolist()
xy_cols = x_cols + y_cols
data = data[xy_cols]
print(data.columns)

if COLNAMES_TO_FLIKE:
    data.columns = [f"f{i}" for i in range(len(data.columns))]
    y_cols = target_names = [data.columns[-1]]

x_cols = feature_names = data.columns.drop(y_cols).tolist()
xy_cols = data.columns.tolist()


# %% Preprocessing -----------------------------------------------------------

XY, col_types_dict, category_dict, float_scaler = preprocess_by_column(data)

X = source_df = XY[x_cols]
Y = target_df = XY[y_cols]
Y_series = target_series = Y[Y.columns[0]]

print(target_names)


y_label_mapping_dict = {
    2: 0,
    0: 1,
    1: 2,
}
y_label_inverse_mapping_dict = {
    0: 2,
    1: 0,
    2: 1,
}
y_class_name_dict = {
    0: 'A Grade',
    1: 'B Grade',
    2: 'C Grade',
}

y_label_encode_list = [
    item[1]
    for item in sorted(list(y_label_mapping_dict.items()), key=lambda x: x[0])
]
y_label_inverse_encode_list = [
    item[1]
    for item in sorted(list(y_label_inverse_mapping_dict.items()), key=lambda x: x[0])
]
y_class_code_list = [
    item[0]
    for item in sorted(list(y_class_name_dict.items()), key=lambda x: x[0])
]
y_class_name_list = [
    item[1]
    for item in sorted(list(y_class_name_dict.items()), key=lambda x: x[0])
]

# y_class_dict = {
#     2: (0, 'A Grade'),
#     0: (1, 'B Grade'),
#     1: (2, 'C Grade'),
# }
# y_class_code_list = [
#     item[1][0]
#     for item in sorted(list(y_class_dict.items()), key=lambda x: x[0])
# ]
# y_class_name_list = [
#     item[1][1]
#     for item in sorted(list(y_class_dict.items()), key=lambda x: x[0])
# ]
# reordered_y_class_list = y_class_code_list


# %% Load a pre-trained Model ------------------------------------------------

model = xgb.XGBClassifier(objective='multi:softprob')
model.load_model(DUMP_PATH)
model.n_classes_ = len(np.unique(Y.values))
# model._le = IntLabelEncoder().fit(y_label_encode_list)
model._le = LabelEncoder().fit(y_label_encode_list)

print(model)
print('X: ', X.shape)
print('Y unique: ', len(np.unique(Y)), np.unique(Y))
print('predict_proba: ', model.predict_proba(X).shape)
print('predict: ', model.predict(X).shape)

# %% TEST Headline -----------------------------------------------------------

from sklearn.metrics import accuracy_score

# %%

y_label_mapping_dict = {
    2: 0,
    0: 1,
    1: 2,
}
y_label_inverse_mapping_dict = {
    0: 2,
    1: 0,
    2: 1,
}
y_class_name_dict = {
    0: 'A Grade',
    1: 'B Grade',
    2: 'C Grade',
}

y_label_encode_list = [
    item[1]
    for item in sorted(list(y_label_mapping_dict.items()), key=lambda x: x[0])
]
y_label_inverse_encode_list = [
    item[1]
    for item in sorted(list(y_label_inverse_mapping_dict.items()), key=lambda x: x[0])
]
y_class_code_list = [
    item[0]
    for item in sorted(list(y_class_name_dict.items()), key=lambda x: x[0])
]
y_class_name_list = [
    item[1]
    for item in sorted(list(y_class_name_dict.items()), key=lambda x: x[0])
]

# model._le = LabelEncoder().fit(y_label_inverse_encode_list)
pred_y = model.predict(X)
real_y = Y_series.values


pred_y_class_dict = {
    2: (0, 'A Grade'),
    0: (1, 'B Grade'),
    1: (2, 'C Grade'),
}

real_y_class_dict = {
    0: (2, 'C Grade'),
    1: (0, 'A Grade'),
    2: (1, 'B Grade'),
}


def change_y_class_int_with_dict(y_arr, y_class_mapping_dict):

    changed_pred_y_arr = np.fromiter(
        (y_class_mapping_dict[v][0] for v in y_arr),
        dtype='int8',
    )

    return changed_pred_y_arr


changed_pred_y = change_y_class_int_with_dict(
    pred_y,
    pred_y_class_dict,
)


changed_real_y = change_y_class_int_with_dict(
    Y_series.values,
    real_y_class_dict,
)

pred_y[:10]
# array([1, 0, 0, 2, 0, 1, 0, 0, 0, 0])

real_y[:10]
# array([0, 1, 0, 1, 1, 2, 1, 1, 1, 1], dtype=int8)

accuracy_score(pred_y[:10], real_y[:10])

changed_pred_y[:10]
changed_real_y[:10]
# array([2, 1, 1, 0, 1, 2, 1, 1, 1, 1], dtype=int8)


pred_y[:10]
accuracy_score(changed_pred_y[:10], real_y[:10])
accuracy_score(pred_y[:10], changed_real_y[:10])

y_label_unique_list = [0, 1, 2]
tle = LabelEncoder()
tle.fit(y_label_unique_list)
tle.classes_

y_label_encode_list
lle = LabelEncoder()
lle.fit(y_label_encode_list)
lle.classes_

y_label_inverse_encode_list
ile = LabelEncoder()
ile.fit(y_label_inverse_encode_list)
ile.classes_


pred_y[:10]
pred_y[:10]
tle.inverse_transform(pred_y[:10])
lle.inverse_transform(pred_y[:10])
ile.inverse_transform(pred_y[:10])

tle.transform(pred_y[:10])
lle.transform(pred_y[:10])
ile.transform(pred_y[:10])

real_y[:10]

changed_pred_y[:10]
accuracy_score(pred_y[:10], real_y[:10])
# %% Test 1
np.array([2, 1, 6], dtype='str')
# le = LabelEncoder()


# %% ------------------------------------------------------


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


# %%

le = IntLabelEncoder().fit([2, 1, 6])
le.classes_
le.transform([6, 6, 2, 1])
le.inverse_transform([0, 2, 0, 1])

# %%


tmp_y_class_dict = {
    2: 0,
    0: 1,
    1: 2,
}

changed_pred_y = np.fromiter(
    (tmp_y_class_dict[v] for v in pred_y),
    dtype='int8',
)

accuracy_score(changed_pred_y, real_y)


# %%
# %%
xgb.plot_importance(model)
xgb.to_graphviz(model)
