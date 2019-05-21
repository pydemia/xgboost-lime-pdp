"""Test Loading XGBoost and Plotting via lime.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


os.chdir('../git/xgboost-lime-pdp')

DUMP_PATH = 'data/nativeBoost2'
# %%

def num_missing(x):
    return sum(x.isnull())

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# %%

train_filename = 'datasets/train_63qYitG.csv'
test_filename = 'datasets/test_XaoFywY.csv'
train_df = pd.read_csv(train_filename, header=0)
test_df = pd.read_csv(test_filename, header=0)
cols=train_df.columns
cols

# %%

train_df['source']='train'
test_df['source']='test'
data = pd.concat([train_df, test_df],ignore_index=True)

# Handling missing values
imputer_mean = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_median = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_mode = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)


data["Life_Style_Index"]=imputer_mean.fit_transform(data[["Life_Style_Index"]]).ravel()
data["Var1"]=imputer_mean.fit_transform(data[["Var1"]]).ravel()
data["Customer_Since_Months"]=imputer_median.fit_transform(data[["Customer_Since_Months"]]).ravel()

X = pd.DataFrame(data)
data = DataFrameImputer().fit_transform(X)
print (data.apply(num_missing, axis=0))

#Divide into test and train:
train_df = data.loc[data['source']=="train"]
test_df = data.loc[data['source']=="test"]


# Drop unwanted columns
train_df = train_df.drop(['Trip_ID','Cancellation_Last_1Month','Confidence_Life_Style_Index','Gender','Life_Style_Index','Var1','Var2','source',],axis=1)
#### Extract the label column
train_target = np.ravel(np.array(train_df['Surge_Pricing_Type'].values))
train_df = train_df.drop(['Surge_Pricing_Type'],axis=1)


# Extract features
float_columns=[]
cat_columns=[]
int_columns=[]

for i in train_df.columns:
    if train_df[i].dtype == 'float' :
        float_columns.append(i)
    elif train_df[i].dtype == 'int64':
        int_columns.append(i)
    elif train_df[i].dtype == 'object':
        cat_columns.append(i)

train_cat_features = train_df[cat_columns]
train_float_features = train_df[float_columns]
train_int_features = train_df[int_columns]

## Transformation of categorical columns
# Label Encoding:
#train_cat_features_ver2 = pd.get_dummies(train_cat_features, columns=['Destination_Type','Type_of_Cab'])
train_cat_features_ver2 = train_cat_features.apply(LabelEncoder().fit_transform)
train_cat_features_ver2['Type_of_Cab'][2]

train_cat_features_ver2.dtypes
## Transformation of float columns
# Rescale data (between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))

for i in train_float_features.columns:
    X_temp = train_float_features[i].values.reshape(-1,1)
    train_float_features[i] = scaler.fit_transform(X_temp)

#### Finalize X & Y
temp_1 = np.concatenate((train_cat_features_ver2,train_float_features),axis=1)
train_transformed_features = np.concatenate((temp_1,train_int_features),axis=1)
train_transformed_features = pd.DataFrame(data=train_transformed_features)

array = train_transformed_features.values
number_of_features = len(array[0])
X = array[:,0:number_of_features]
Y = train_target

# Split into training and validation set

from sklearn.model_selection import train_test_split
validation_size = 0.2
seed = 7
# X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.33, random_state=42)


scoring = 'accuracy'

# LIME SECTION
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
#from __future__ import print_function

# Line-up the feature names
feature_names_cat = list(train_cat_features_ver2)
feature_names_float = list(train_float_features)
feature_names_int = list(train_int_features)
feature_names_cat

feature_names = sum([feature_names_cat, feature_names_float, feature_names_int], [])
print(feature_names)


# %%
# model = xgb.Booster()
# model.load_model(DUMP_PATH)

# %%
#
# model = xgb.XGBClassifier()
# model.load_model(DUMP_PATH)

# # %%
model = xgb.XGBModel()
model.load_model(DUMP_PATH)
# %%
X_train.shape
feature_names
cat_columns
feature_names_cat
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['1', '2', '3'],
    categorical_features=cat_columns,
    categorical_names=feature_names_cat, kernel_width=3
)
# pred_prob_fn =

# Pick the observation in the validation set for which explanation is required
observation_1 = 2

# Get the explanation for Logistic Regression
exp = explainer.explain_instance(
    X_validation[observation_1],
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
