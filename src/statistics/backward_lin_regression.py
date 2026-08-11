from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_feature_differences(question, feature_df, order: list = ["A", "B"]):
    X_features = feature_df.loc[question["X"]].values
    H_features = feature_df.loc[question[order[0]]].values
    L_features = feature_df.loc[question[order[1]]].values
    XH_diff = X_features - H_features
    XL_diff = X_features - L_features

    return abs(XL_diff) - abs(XH_diff)


def backward_stepwise_regression(
    y,
    feature_df: pd.DataFrame = None,
    questions_df: pd.DataFrame = None,
    X=None,
    feature_names=None,
    alpha=0.05,
    order_features_by: Literal["coef", "std_err", "p"] = None,
):
    """
    Perform backward stepwise regression
    """

    if feature_names is None:
        feature_names = feature_df.columns
    if X is None:
        X = np.stack(questions_df.apply(lambda x: get_feature_differences(x, feature_df), axis=1))

    # Start with all features
    current_features = list(range(X.shape[1]))
    current_pvalues = None
    removed_features = []

    while True:
        # Fit model with current features
        X_current = X[:, current_features]
        X_sm = sm.add_constant(X_current)
        model = sm.OLS(y, X_sm).fit()

        # Get p-values (excluding intercept)
        pvalues = model.pvalues[1:]

        # Find feature with highest p-value
        max_p = pvalues.max()
        max_p_idx = pvalues.argmax()

        if max_p > alpha and len(current_features) > 1:
            # Remove feature
            removed_features.append((current_features[max_p_idx], max_p))
            del current_features[max_p_idx]
        else:
            break

    # Final model
    X_final = X[:, current_features]
    X_sm = sm.add_constant(X_final)
    final_model = sm.OLS(y, X_sm).fit()

    y_pred = final_model.predict(X_sm)

    if order_features_by is not None:
        stat_map = {}
        if order_features_by == "coef":
            stat = model.params[1:]
        elif order_features_by == "std_err":
            stat = model.bse[1:]
        elif order_features_by == "p":
            stat = model.pvalues[1:]
        else:
            stat = None
        if stat is not None:
            sorted_pairs = sorted(
                zip(current_features, stat), key=lambda x: x[1], reverse=(order_features_by == "coef")
            )
            current_features = [idx for idx, _ in sorted_pairs]

    print(final_model.summary())
    print(f"Selected features: {[str(feature_names[i]) for i in current_features]}")

    return final_model, current_features, removed_features, y_pred
