import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    from typing import List, Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import opensmile
    import pandas as pd
    import seaborn as sns
    import torch
    import torchaudio
    from sklearn.preprocessing import StandardScaler
    from speechbrain.inference.encoders import MelSpectrogramEncoder

    from src import load_singer_identity_model
    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        PLOT_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
    )
    from src.statistics.feature_correlation import (
        get_all_distance_differences,
        get_distance_row,
        get_global_distance_scores,
        scale_df,
    )
    from src.statistics.lin_regression import get_feature_differences
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_heatmap,
        plot_correlation_scatter,
        plot_feature_connection,
    )
    from src.survey_dataset_helpers import load_survey_data
    from src.utils import get_trimmed_audio

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        get_all_distance_differences,
        get_feature_differences,
        get_trimmed_audio,
        load_survey_data,
        mo,
        np,
        opensmile,
        os,
        pd,
        plot_correlation_bar,
        plot_correlation_heatmap,
        plot_feature_connection,
        scale_df,
    )


@app.cell
def _(CSV_FOLDER, DATASET_FOLDER, os):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey", "survey_2")
    CSV_PATHS = {
        "participants": os.path.join(SURVEY_FOLDER, "participants.csv"),
        "songs": os.path.join(SURVEY_FOLDER, "songs.csv"),
        "answers": os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"),
        "questions": os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"),
        "tracks": os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "dataset_survey_2_final.csv",
        ),
    }
    return (CSV_PATHS,)


@app.cell
def _(PLOT_FOLDER, os):
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2")
    return (PLOT_SAVE_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load Dataset
    """)
    return


@app.cell
def _(CSV_PATHS, load_survey_data):
    SURVEY_DATA = load_survey_data(CSV_PATHS)
    questions_df = SURVEY_DATA["questions_df"]
    answers_df = SURVEY_DATA["answers_df"]
    participants_df = SURVEY_DATA["participants_df"]
    songs_df = SURVEY_DATA["songs_df"]
    human_agreement = SURVEY_DATA["human_agreement"]
    answer_a_b_ratio = SURVEY_DATA["answer_a_b_ratio"]
    track_df = SURVEY_DATA["track_df"]
    return questions_df, songs_df, track_df


@app.cell
def _(get_all_distance_differences, questions_df, scale_df, track_df):
    hl_features = [
        "pred_genre_main",
        "pred_genre_sub",
        "pred_approachability",
        "pred_danceable",
        "pred_not_danceable",
        "pred_engagement",
        "pred_mood_and_theme",
        "pred_tempo",
        "pred_gender",
        "pred_p_male",
        "pred_p_female",
        "pred_age",
        "pred_age_no_trim",
    ]
    scaled_track_df = scale_df(
        track_df,
        [
            "pred_approachability",
            "pred_danceable",
            "pred_not_danceable",
            "pred_engagement",
            "pred_tempo",
            "pred_p_male",
            "pred_p_female",
            "pred_age",
            "pred_age_no_trim",
        ],
    )
    hl_distances = get_all_distance_differences(
        scaled_track_df, hl_features, questions_df
    )
    hl_distances
    return


@app.cell
def _(mo):
    mo.md(r"""
    # GeMAPS Feature Set
    """)
    return


@app.cell
def _(opensmile):
    smile_gemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return (smile_gemaps,)


@app.cell
def _(SAMPLE_RATE, get_trimmed_audio):
    def get_feature_set(song_path, sm):
        trimmed_audio = get_trimmed_audio(song_path, sr=SAMPLE_RATE)
        return sm.process_signal(trimmed_audio, SAMPLE_RATE).values[0]

    return


@app.cell
def _(DATASET_FOLDER, os):
    gemaps_feature_path = os.path.join(
        DATASET_FOLDER, "fma_large_feature_sets", "survey_2_gemaps.npy"
    )
    return (gemaps_feature_path,)


@app.cell
def _():
    """
    gemaps_features = pd.DataFrame(
        track_df.song_path.apply(
            lambda x: get_feature_set(x, smile_gemaps)
        ).tolist(),
        columns=smile_gemaps.feature_names,
        index=track_df.index,
    )
    gemaps_features


    with open(gemaps_feature_path, "wb") as npyfile:
        np.save(npyfile, gemaps_features.values)
    """
    return


@app.cell
def _(gemaps_feature_path, np, pd, scale_df, smile_gemaps, track_df):
    gemaps_features_df = pd.DataFrame(
        np.load(gemaps_feature_path),
        columns=smile_gemaps.feature_names,
        index=track_df.index,
    )
    gemaps_features_df = scale_df(gemaps_features_df)
    gemaps_features_df
    return (gemaps_features_df,)


@app.cell
def _(gemaps_features_df, get_all_distance_differences, questions_df):
    gemaps_distances = get_all_distance_differences(
        gemaps_features_df, gemaps_features_df.columns, questions_df
    )
    gemaps_distances
    return (gemaps_distances,)


@app.cell
def _(gemaps_distances, plot_correlation_bar, questions_df):
    top_correlating_gemaps_features = plot_correlation_bar(
        title="GeMAPS Feature Correlations (All)",
        feature_df=gemaps_distances,
        target_feature=questions_df["A_perc"],
        top_x=20,
        output=True,
    )
    return (top_correlating_gemaps_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regression Analysis
    """)
    return


@app.cell
def _():
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score, mean_squared_error

    return Lasso, mean_squared_error, r2_score


@app.cell
def _(gemaps_features_df, get_feature_differences, np, questions_df):
    # prepare H and L voices
    lr_sim_y = (
        questions_df["A_perc"].apply(lambda x: x if x >= 0.5 else 1 - x).values
        - 0.5
    ) * 2

    lr_X = np.stack(
        [
            get_feature_differences(
                q,
                gemaps_features_df,
                ["A", "B"] if q["A_perc"] >= 0.5 else ["B", "A"],
            )
            for i, (_, q) in enumerate(questions_df.iterrows())
        ]
    )
    return lr_X, lr_sim_y


@app.cell
def _(Lasso, gemaps_features_df, lr_X, lr_sim_y, pd):
    sim_model = Lasso(alpha=0.01, positive=True, max_iter=10000)

    sim_model.fit(lr_X, lr_sim_y)

    y_pred_sim = sim_model.predict(lr_X)

    sim_coef = pd.Series(
        sim_model.coef_, index=gemaps_features_df.columns
    ).sort_values(ascending=False)

    sim_coef = sim_coef[sim_coef > 0]
    sim_coef
    return sim_coef, sim_model, y_pred_sim


@app.cell
def _(sim_model):
    sim_model.intercept_
    return


@app.cell
def _(lr_sim_y, mean_squared_error, np, r2_score, y_pred_sim):
    print(f"R2 score: {r2_score(lr_sim_y, y_pred_sim):.3f}")
    print(f"RMSE    : {np.sqrt(mean_squared_error(lr_sim_y, y_pred_sim)):.3f}")
    return


@app.cell
def _(plot_feature_connection, sim_coef, top_correlating_gemaps_features):
    plot_feature_connection(
        set_1=sim_coef.index.values,
        set_2=top_correlating_gemaps_features[0],
        set_1_label="Similarity Ratings",
        set_2_label="Top Correlations",
        top_x=19,
        title="Lasso Predictors Comparison",
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_features_df,
    os,
    plot_correlation_heatmap,
    sim_coef,
):
    plot_correlation_heatmap(
        gemaps_features_df[sim_coef.index.values],
        "Pairwise Pearson Correlation Coefficients (r)",
        save_path=os.path.join(
            PLOT_SAVE_DIR, "gemaps_lasso_pairwise_correlation.png"
        ),
        labelsize=12,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gender Lasso
    """)
    return


@app.cell
def _(gemaps_features_df):
    gemaps_features_df.values.shape
    return


@app.cell
def _(songs_df):
    songs_df[songs_df.skipInSurvey].trackID.values
    return


@app.cell
def _(songs_df, track_df):
    skip_mask = ~track_df.index.isin(songs_df[songs_df.skipInSurvey].trackID.values)
    return (skip_mask,)


@app.cell
def _(skip_mask, track_df):
    track_df[skip_mask]
    return


@app.cell
def _(Lasso, gemaps_features_df, pd, skip_mask, track_df):
    lr_X_gender = gemaps_features_df[skip_mask].values
    lr_gender_y = track_df[skip_mask]["pred_p_male"]

    gender_model = Lasso(alpha=0.03, max_iter=10000)

    gender_model.fit(lr_X_gender, lr_gender_y)

    y_gender_pred = gender_model.predict(lr_X_gender)

    gender_coef = pd.Series(
        gender_model.coef_, index=gemaps_features_df.columns
    )
    gender_coef = gender_coef.reindex(
        gender_coef.abs().sort_values(ascending=False).index
    )

    # gender_coef = gender_coef[gender_coef > 0]
    gender_coef[abs(gender_coef) > 0]
    return gender_coef, gender_model, lr_gender_y, y_gender_pred


@app.cell
def _(gender_model):
    gender_model.intercept_
    return


@app.cell
def _(lr_gender_y, mean_squared_error, np, r2_score, y_gender_pred):
    print(f"R2 score: {r2_score(lr_gender_y, y_gender_pred):.3f}")
    print(
        f"RMSE    : {np.sqrt(mean_squared_error(lr_gender_y, y_gender_pred)):.3f}"
    )
    return


@app.cell
def _(gender_coef, plot_feature_connection, sim_coef):
    plot_feature_connection(
        set_1=sim_coef.index.values,
        set_2=gender_coef.index.values,
        set_1_label="Similarity Ratings",
        set_2_label="Gender Regression",
        top_x=19,
        title="Lasso Predictors Comparison",
    )
    return


if __name__ == "__main__":
    app.run()
