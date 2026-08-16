import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    def # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    from typing import List, Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
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
    )
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_scatter,
    )
    from src.survey_dataset_helpers import load_survey_data
    from src.utils import get_trimmed_audio

    return (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        get_all_distance_differences,
        get_global_distance_scores,
        load_survey_data,
        mo,
        os,
        plot_correlation_bar,
        plot_correlation_scatter,
        scale_df,
    )


@app.cell
def _(AUDIO_FOLDER, CSV_FOLDER, DATASET_FOLDER, os):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey", "survey_2")
    CSV_PATHS = {
        "participants": os.path.join(SURVEY_FOLDER, "participants.csv"),
        "songs": os.path.join(SURVEY_FOLDER, "songs.csv"),
        "answers": os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"),
        "questions": os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"),
        "song_files": os.path.join(AUDIO_FOLDER, "fma_large"),
        "stem_files": os.path.join(AUDIO_FOLDER, "fma_large_stems"),
        "tracks": os.path.join(
            CSV_FOLDER,
            "LargeDataset",
            "dataset_survey_2_final.csv",
        ),
    }
    return (CSV_PATHS,)


@app.cell
def _(PLOT_FOLDER, os):
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2", "03_01")
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
    return questions_df, track_df


@app.cell
def _(mo):
    mo.md(r"""
    # Single High Level Feature Agreement Scores
    """)
    return


@app.cell
def _():
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
    return (hl_features,)


@app.cell
def _(
    get_all_distance_differences,
    hl_features,
    questions_df,
    scale_df,
    track_df,
):
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
    return (hl_distances,)


@app.cell
def _(PLOT_SAVE_DIR, hl_distances, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="High-Level Feature Correlations",
        feature_df=hl_distances,
        target_feature=questions_df["A_perc"],
        top_x=len(hl_distances.columns) - 1,
        save_path=os.path.join(
            PLOT_SAVE_DIR, "High-Level Feature Correlations (Individual).png"
        ),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, hl_distances, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="High Level Feature Correlations (Randomized)",
        feature_df=hl_distances[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=len(hl_distances.columns),
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "High-Level Feature Correlations (Individual, Randomized).png",
        ),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, hl_distances, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="High Level Feature Correlations (Heuristic)",
        feature_df=hl_distances[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=len(hl_distances.columns),
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "High-Level Feature Correlations (Individual, Heuristic).png",
        ),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, hl_distances, os, plot_correlation_scatter, questions_df):
    plot_correlation_scatter(
        feature_name="Gender",
        x=questions_df["A_perc"],
        y=hl_distances["pred_gender"],
        save_path=os.path.join(
            PLOT_SAVE_DIR, "Gender Feature Correlation Scatter.png"
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # High Level Feature Set
    """)
    return


@app.cell
def _(get_global_distance_scores, questions_df, track_df):
    hl_feature_distances_df = get_global_distance_scores(
        track_df[
            [
                "pred_approachability",
                "pred_danceable",
                "pred_engagement",
                "pred_tempo",
                "pred_p_male",
                "pred_age",
            ]
        ],
        questions_df,
    )
    return (hl_feature_distances_df,)


@app.cell
def _(
    PLOT_SAVE_DIR,
    hl_feature_distances_df,
    os,
    plot_correlation_bar,
    questions_df,
):
    plot_correlation_bar(
        title="High Level Feature Set Correlations (All)",
        feature_df=hl_feature_distances_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
        save_path=os.path.join(
            PLOT_SAVE_DIR, "High-Level Features Correlations (All).png"
        ),
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    hl_feature_distances_df,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="High Level Feature Set (Canberra)",
        feature_name="High_Level_Feature_Set_Canberra",
        y=hl_feature_distances_df["distance_canberra"],
        x=questions_df["A_perc"],
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "High-Level Features Correlation Scatter (Canberra Distance).png",
        ),
    )
    return


if __name__ == "__main__":
    app.run()
