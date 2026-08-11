from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_feature_differences(question, feature_df, order: list = ["A", "B"]):
    X_features = feature_df.loc[question["X"]].values
    M_features = feature_df.loc[question[order[0]]].values
    L_features = feature_df.loc[question[order[1]]].values
    XM_diff = X_features - M_features
    XL_diff = X_features - L_features

    return abs(XL_diff) - abs(XM_diff)
