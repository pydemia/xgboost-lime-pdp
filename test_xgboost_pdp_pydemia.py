"""Test Loading XGBoost and Plotting via pdpbox.
"""


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import pdpbox
from pdpbox import pdp, info_plots


print(xgb.__version__)
print(pdpbox.__version__)

os.chdir('../git/xgboost-lime-pdp')


# DUMP_PATH = 'data/nativeBoost2'
# train_filename = 'data/train_63qYitG.csv'
# test_filename = 'data/test_XaoFywY.csv'

DUMP_PATH = 'data/nativeBoost6features'
train_filename = 'data/train_new.csv'
test_filename = 'data/test_new.csv'

X = train_df = pd.read_csv(train_filename, header=0)
Y = test_df = pd.read_csv(test_filename, header=0)
XY = pd.concat([X, Y], axis=1)

x_cols = feature_names = X.columns.tolist()
y_cols = Y.columns.tolist()
xy_cols = XY.columns.tolist()




model = xgb.XGBModel()
model.load_model(DUMP_PATH)

_tmp_model = xgb.XGBClassifier()
_tmp_model.load_model(DUMP_PATH)
_tmp_model.predict_proba(X).shape

# %% Recommended Method to load a proper model: `LabelEncoder` ---------------

from sklearn.preprocessing import LabelEncoder


clf = xgb.XGBClassifier()
booster = xgb.Booster()
booster.load_model(DUMP_PATH)
clf._Booster = booster
clf._le = LabelEncoder().fit(Y)

# %% Test ----
y_pred_proba = clf.predict_proba(X)
y_pred = clf.predict(X)

y_pred_proba.shape
y_pred.shape
# %%
# %%
model.predict(X).shape
# %%

# %%
model.predict(X).shape
# %%
x_cols
# %%
# model._Booster.feature_names = [for model._Booster.feature_names]
# aa[-1]
# %%
# X.index = [f'f{i}' for i in X.index]
# Y.index = [f'f{i}' for i in Y.index]
# %%
print(X.shape, Y.shape)
# %%
x_cols = X.columns.tolist()
y_cols = Y.columns.tolist()

# %%

# default
fig, axes, summary_df = info_plots.target_plot(
    df=XY,
    feature=x_cols[2],
    feature_name=x_cols[2],
    target=y_cols[0],
)

# %%

fig, axes, df = info_plots.actual_plot(
    model=clf,
    X=X,
    feature=x_cols[1],
    feature_name=x_cols[1],
    which_classes=[0, 3, 6],
    predict_kwds={},  # This parameter should be passed to avoid a strange TypeError
)
# %%
pdp_isolated_tmp = pdp.pdp_isolate(
    model=clf,
    dataset=X,
    model_features=x_cols,
    feature=x_cols[0],
    n_jobs=1,
)

# %%
fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_isolated_tmp,
    feature_name=x_cols[:2],
    center=True, x_quantile=True,
    ncols=3, plot_lines=True, frac_to_plot=100,
    plot_pts_dist=True,
)
# %%
# default
fig, axes, summary_df = info_plots.target_plot_interact(
    df=XY,
    features=x_cols[2:],
    feature_names=x_cols[2:],
    target=y_cols[0],
)

# %%
fig, axes, summary_df = info_plots.actual_plot_interact(
    model=clf,
    X=X,
    features=x_cols[3:],
    feature_names=x_cols[3:],
    which_classes=[2, 5],
)
# %%
pdp_interacted_tmp= pdp.pdp_interact(
    model=clf,
    dataset=X,
    model_features=x_cols,
    features=x_cols[:2],
    num_grid_points=[10, 10],
    percentile_ranges=[None, None],
    n_jobs=1,
)
# %%
fig, axes = pdp.pdp_interact_plot(
    pdp_interacted_tmp,
    feature_names=x_cols,
    plot_type='grid',
    x_quantile=True,
    ncols=2,
    plot_pdp=True,
    which_classes=[1, 2, 3],
)

# %%
fig, axes = pdp.pdp_interact_plot(
    pdp_interacted_tmp,
    feature_names=x_cols,
    plot_type='contour',
    x_quantile=True,
    ncols=1,
    plot_pdp=True,
    which_classes=[1, 2],
)



# %% Hand-Made Debugging -----------------------------------------------------

from joblib import Parallel, delayed
dataset = X
model_features = x_cols
feature = x_cols[0]
grid_type = 'percentile'
percentile_range = None
memory_limit = .5
num_grid_points = 20
grid_range = None
cust_grid_points = None
data_transformer = None
predict_kwds = {}
n_jobs=1

print(model_features)
print(feature)

# %%

def _calc_ice_lines(feature_grid, data, model, model_features, n_classes, feature, feature_type,
                    predict_kwds, data_transformer, unit_test=False):
    """Apply predict function on a feature_grid

    Returns
    -------
    Predicted result on this feature_grid
    """

    _data = data.copy()
    if feature_type == 'onehot':
        # for onehot encoding feature, need to change all levels together
        other_grids = [grid for grid in feature if grid != feature_grid]
        _data[feature_grid] = 1
        for grid in other_grids:
            _data[grid] = 0
    else:
        _data[feature] = feature_grid

    # if there are other features highly depend on the investigating feature
    # other features should also adjust based on the changed feature value
    # Example:
    # there are 3 features: a, b, a_b_ratio
    # if feature a is the investigated feature
    # data_transformer should be:
    # def data_transformer(df):
    #   df["a_b_ratio"] = df["a"] / df["b"]
    #   return df
    if data_transformer is not None:
        _data = data_transformer(_data)

    if n_classes == 0:
        predict = model.predict
    else:
        predict = model.predict_proba

    # get predictions for this chunk
    preds = predict(_data[model_features], **predict_kwds)

    if n_classes == 0:
        grid_results = pd.DataFrame(preds, columns=[feature_grid])
    elif n_classes == 2:
        grid_results = pd.DataFrame(preds[:, 1], columns=[feature_grid])
    else:
        grid_results = []
        for n_class in range(n_classes):
            grid_result = pd.DataFrame(preds[:, n_class], columns=[feature_grid])
            grid_results.append(grid_result)

    # _data is returned for unit test
    if unit_test:
        return grid_results, _data
    else:
        return grid_results


n_classes, predict = pdpbox.utils._check_model(model)
pdpbox.utils._check_dataset(df=dataset)
_dataset = X.copy()

feature_type = pdpbox.utils._check_feature(feature=feature, df=_dataset)
feature_type
pdpbox.utils._check_grid_type(grid_type=grid_type)
pdpbox.utils._check_percentile_range(percentile_range=percentile_range)
pdpbox.utils._check_memory_limit(memory_limit=memory_limit)

percentile_info = []
if feature_type == 'binary':
    feature_grids = np.array([0, 1])
    display_columns = ['%s_0' % feature, '%s_1' % feature]
elif feature_type == 'onehot':
    feature_grids = np.array(feature)
    display_columns = feature
else:
    # calculate grid points for numeric features
    if cust_grid_points is None:
        feature_grids, percentile_info = pdpbox.utils._get_grids(
            feature_values=_dataset[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
            percentile_range=percentile_range, grid_range=grid_range)
    else:
        # make sure grid points are unique and in ascending order
        feature_grids = np.array(sorted(np.unique(cust_grid_points)))
    display_columns = [pdpbox.utils._get_string(v) for v in feature_grids]

feature_grids
display_columns

# %%
"""
Real-Code: Parallel Genereator Expression

grid_results = Parallel(n_jobs=true_n_jobs)(
    delayed(_calc_ice_lines)(
        feature_grid, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
        feature=feature, feature_type=feature_type, predict_kwds=predict_kwds, data_transformer=data_transformer)
    for feature_grid in feature_grids
)

The Following: Converting as For-Loop to track variables.
"""
# Parallel calculate ICE lines
final_grid_results = []
for __feature_grid in feature_grids:
    # res = _calc_ice_lines(
    #     feature_grid, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
    #     feature=feature, feature_type=feature_type, predict_kwds=predict_kwds, data_transformer=data_transformer,
    # )
    __data = _dataset
    __model = model
    __model_features = model_features
    __n_classes = n_classes
    __feature = feature
    __feature_type = feature_type
    __predict_kwds = predict_kwds
    __data_transformer = data_transformer

    _data = __data.copy()
    if __feature_type == 'onehot':
        # for onehot encoding feature, need to change all levels together
        other_grids = [grid for grid in __feature if grid != __feature_grid]
        __data[__feature_grid] = 1
        for grid in other_grids:
            _data[grid] = 0
    else:
        __data[feature] = __feature_grid

    # if there are other features highly depend on the investigating feature
    # other features should also adjust based on the changed feature value
    # Example:
    # there are 3 features: a, b, a_b_ratio
    # if feature a is the investigated feature
    # data_transformer should be:
    # def data_transformer(df):
    #   df["a_b_ratio"] = df["a"] / df["b"]
    #   return df
    if __data_transformer is not None:
        __data = __data_transformer(__data)

    if __n_classes == 0:
        predict = __model.predict
    else:
        predict = __model.predict_proba

    # get predictions for this chunk
    preds = predict(__data[__model_features], **__predict_kwds)

    if __n_classes == 0:
        grid_results = pd.DataFrame(preds, columns=[__feature_grid])
    elif __n_classes == 2:
        grid_results = pd.DataFrame(preds[:, 1], columns=[__feature_grid])
    else:
        grid_results = []
        for n_class in range(__n_classes):
            grid_result = pd.DataFrame(
                preds[:, n_class], columns=[__feature_grid],
            )
            grid_results.append(grid_result)

    res_grid_results = grid_results
    grid_results += [res_grid_results]

# %%

__data.shape
__model_features
preds.shape

feature_grids
__feature_grid
[__feature_grid]
