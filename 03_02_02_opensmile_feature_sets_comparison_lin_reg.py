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
        get_global_distance_scores,
        scale_df,
        get_distance_row,
    )
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_scatter,
        plot_feature_connection,
        plot_correlation_heatmap,
    )
    from src.statistics.backward_lin_regression import (
        get_feature_differences,
        backward_stepwise_regression,
    )
    from src.survey_dataset_helpers import load_survey_data
    from src.utils import get_trimmed_audio

    return (
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        StandardScaler,
        get_all_distance_differences,
        get_feature_differences,
        get_trimmed_audio,
        load_survey_data,
        mo,
        np,
        opensmile,
        os,
        pd,
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
    return


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
    return questions_df, track_df


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regression Analysis
    """)
    return


@app.cell
def _():
    from scipy.optimize import nnls
    from sklearn.linear_model import Lasso

    return (Lasso,)


@app.cell
def _(
    StandardScaler,
    gemaps_features_df,
    get_feature_differences,
    np,
    questions_df,
):
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

    scaler = StandardScaler()
    lr_X_s = scaler.fit_transform(lr_X)
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
    return (sim_coef,)


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
def _(Lasso, gemaps_features_df, pd, track_df):
    lr_X_gender = gemaps_features_df.values
    lr_gender_y = track_df["pred_p_male"]

    gender_model = Lasso(alpha=0.01, positive=True, max_iter=10000)

    gender_model.fit(lr_X_gender, lr_gender_y)

    y_pred = gender_model.predict(lr_X_gender)

    gender_coef = pd.Series(
        gender_model.coef_, index=gemaps_features_df.columns
    ).sort_values(ascending=False)

    gender_coef = gender_coef[gender_coef > 0]
    gender_coef
    return (gender_coef,)


@app.cell
def _(sim_coef):
    len(sim_coef)
    return


@app.cell
def _(gender_coef, plot_feature_connection, sim_coef):
    plot_feature_connection(
        set_1=sim_coef.index.values,
        set_2=gender_coef.index.values,
        set_1_label="Similarity Ratings",
        set_2_label="Gender Regression",
        top_x=len(sim_coef),
    )
    return


if __name__ == "__main__":
    app.run()
