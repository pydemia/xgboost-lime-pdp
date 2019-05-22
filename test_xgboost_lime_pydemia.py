"""Test Loading XGBoost and Plotting via lime.
"""


import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import sklearn as skl
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import xgboost as xgb
# from xgboost.compat import XGBLabelEncoder
# => is IDENTICAL to `sklearn.preprocessing.LabelEncoder`

import lime
from lime import lime_tabular


for _pkg in [np, pd, skl, xgb, mpl]:
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
    If a column has `str`, it will be categorized & substitute by int,
    which is its code number.
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
            XY
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

    XY = data

    # XY.columns = [f"f{i}" for i in range(len(XY.columns))]
    # X.columns = XY.columns[:len(X.columns)]
    # Y.columns = XY.columns[len(X.columns):]

    y_cols = target_names = ['Surge_Pricing_Type']
    x_cols = feature_names = data.columns.drop(target_names).tolist()
    xy_cols = data.columns.tolist()

elif int(CASE) == 2:
    int_type_colnames = ['c1', 'c2', 'c6', 'r1']
    col_converter = {colname: as_int_str for colname in int_type_colnames}

    X = train_df = pd.read_csv(
        train_filename, header=0, converters=col_converter,
    )
    Y = test_df = pd.read_csv(
        test_filename, header=0, converters=col_converter,
    )
    XY = data = pd.concat([X, Y], axis=1)

    XY.columns = [f"f{i}" for i in range(len(XY.columns))]
    X.columns = XY.columns[:len(X.columns)]
    Y.columns = XY.columns[len(X.columns):]

    y_cols = target_names = Y.columns.tolist()
    x_cols = feature_names = X.columns.tolist()
    xy_cols = XY.columns.tolist()


# %% Preprocessing -----------------------------------------------------------

XY, col_types_dict, category_dict, float_scaler = preprocess_by_column(XY)

X = source_df = XY[XY.columns.drop(target_names)]
Y = target_df = XY[target_names]
Y_series = target_series = Y[Y.columns[0]]

print(target_names)


# %% Load a pre-trained Model ------------------------------------------------

model = xgb.XGBClassifier(objective='multi:softprob')
model.load_model(DUMP_PATH)
model.n_classes_ = len(np.unique(Y.values))
model._le = LabelEncoder().fit(np.unique(Y.values))

print(model)
print('X: ', X.shape)
print('Y unique: ', len(np.unique(Y)), np.unique(Y))
print('predict_proba: ', model.predict_proba(X).shape)
print('predict: ', model.predict(X).shape)


# %% Plot: Blackbox Interpretation via `lime` --------------------------------

input_df = X
input_arr = input_df.values
category_colnames = list(category_dict.keys())

explainer = lime.lime_tabular.LimeTabularExplainer(
    input_arr,
    mode='classification',
    feature_names=xy_cols,  # [f"f{i}" for i in range(len(XY.columns))],
    class_names=category_dict[target_names[0]],  # + ['4'],
    categorical_features=category_colnames,
    categorical_names=category_colnames,
    feature_selection='auto',  # 'forward_selection', 'lasso_path', 'auto'
    kernel_width=4,
)

# Get the explanation for Logistic Regression
exp = explainer.explain_instance(
    data_row=input_arr[1],
    predict_fn=model.predict_proba,
    labels=np.unique(Y),
    num_features=5,
    num_samples=3000,
)

# exp_html = exp.as_html()
exp.save_to_file('lime_result.html')

exp.show_in_notebook(show_all=True)

print('finished.')