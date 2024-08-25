import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -4.27083240989936e-05
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=LinearSVR(C=25.0, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.01)),
    XGBRegressor(learning_rate=0.1, max_depth=8, min_child_weight=13, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.8, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
