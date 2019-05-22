"""Test Loading XGBoost and Plotting via lime.
"""


import os
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
import lime
from lime import lime_tabular


os.chdir('../git/xgboost-lime-pdp')

DUMP_PATH = 'data/nativeBoost2'
train_filename = 'data/train_63qYitG.csv'
test_filename = 'data/test_XaoFywY.csv'

# DUMP_PATH = 'data/nativeBoost6features'
# train_filename = 'data/train_new.csv'
# test_filename = 'data/test_new.csv'

train_df = pd.read_csv(train_filename, header=0)
test_df = pd.read_csv(test_filename, header=0)
data = pd.concat(
    [train_df, test_df],
    axis=0,
    ignore_index=True,
)


class DataFrameImputer(TransformerMixin):

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


data = DataFrameImputer().fit_transform(data)

data_cols = data.columns.tolist()

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
data.columns
#### Extract the label column
target_colname = ['Surge_Pricing_Type']
target_series = data['Surge_Pricing_Type']
source_df = data[data.columns.drop(target_colname)]

# %% Column Typing & Categorization ------------------------------------------

col_types = {
    dtype.name: colname_list.tolist()
    for dtype, colname_list
    in source_df.columns.to_series().groupby(source_df.dtypes).groups.items()
}
col_types.keys()

float_columns = col_types['float64']
int_columns = col_types['int64']
str_columns = col_types['object']

source_df[str_columns] = source_df[str_columns].astype('category')
str_codes_df = source_df[str_columns].apply(lambda x: x.cat.codes)
str_names_df = source_df[str_columns].apply(lambda x: x.cat.categorical)
source_df[str_columns] = str_codes_df

scaler = MinMaxScaler(feature_range=(0, 1))

source_df[float_columns] = scaler.fit_transform(source_df[float_columns])

# %% Load a pre-trained Model ------------------------------------------------

model = xgb.XGBModel()
model.load_model(DUMP_PATH)

_tmp_model = xgb.XGBClassifier()
_tmp_model.load_model(DUMP_PATH)
_tmp_model.predict_proba(source_df).shape

# %% Recommended Method to load a proper model: `LabelEncoder` ---------------
#
# from sklearn.preprocessing import LabelEncoder
#
#
# clf = xgb.XGBClassifier()
# booster = xgb.Booster()
# booster.load_model(DUMP_PATH)
# clf._Booster = booster
# clf._le = LabelEncoder().fit(Y)
#
# # %% Test ----
# y_pred_proba = clf.predict_proba(XY)
# y_pred = clf.predict(X)
#
# y_pred_proba.shape
# y_pred.shape
# model.predict(X).shape
# %%

source_arr = source_df.values

explainer = lime.lime_tabular.LimeTabularExplainer(
    source_arr,
    feature_names=source_df.columns.tolist(),
    class_names=['1', '2', '3'],
    categorical_features=str_columns,
    categorical_names=str_columns,
    kernel_width=3,
)

# Get the explanation for Logistic Regression
exp = explainer.explain_instance(
    source_arr[5],
    model.predict,
    num_features=6,
)
exp.show_in_notebook(show_all=False)

# %%
# exp.as_html()

exp.save_to_file(
    'tmp.html',
    labels=None, predict_proba=True, show_predicted_value=True,
)
