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
    from src.statistics.plotting import (
        plot_correlation_bar,
        plot_correlation_heatmap,
        plot_correlation_scatter,
        plot_feature_connection,
    )
    from src.survey_dataset_helpers import load_survey_data
    from src.utils import get_trimmed_audio

    return (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        PLOT_FOLDER,
        get_all_distance_differences,
        get_distance_row,
        get_global_distance_scores,
        get_trimmed_audio,
        load_survey_data,
        mo,
        np,
        opensmile,
        os,
        pd,
        plot_correlation_bar,
        plot_correlation_heatmap,
        plot_correlation_scatter,
        plot_feature_connection,
        plt,
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
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2", "03_02_01")
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
    return answers_df, questions_df, track_df


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
    hl_distances = get_all_distance_differences(scaled_track_df, hl_features, questions_df)
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
    gemaps_feature_path = os.path.join(DATASET_FOLDER, "fma_large_feature_sets", "survey_2_gemaps.npy")
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
    ## Single Feature Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, get_all_distance_differences, questions_df):
    gemaps_distances = get_all_distance_differences(gemaps_features_df, gemaps_features_df.columns, questions_df)
    gemaps_distances
    return (gemaps_distances,)


@app.cell
def _(PLOT_SAVE_DIR, gemaps_distances, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (Randomized)",
        feature_df=gemaps_distances[questions_df.randomized],
        target_feature=questions_df[questions_df.randomized]["A_perc"],
        top_x=10,
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "GeMAPS Feature Correlations (Individual, Randomized).png",
        ),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_distances, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations  (Heuristic)",
        feature_df=gemaps_distances[~questions_df.randomized],
        target_feature=questions_df[~questions_df.randomized]["A_perc"],
        top_x=10,
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "GeMAPS Feature Correlations (Individual, Heuristic).png",
        ),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_distances, os, plot_correlation_bar, questions_df):
    gemaps_single_f_corr = plot_correlation_bar(
        title="Individual GeMAPS Features Correlations",
        feature_df=gemaps_distances,
        target_feature=questions_df["A_perc"],
        top_x=15,
        output=True,
        save_path=os.path.join(PLOT_SAVE_DIR, "GeMAPS Feature Correlations (Individual).png"),
    )
    gemaps_single_f_corr
    return (gemaps_single_f_corr,)


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        feature_name="F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        y=gemaps_distances["F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope"],
        x=questions_df["A_perc"],
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope Feature Correlation Scatter.png",
        ),
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        feature_name="StddevVoicedSegmentLengthSec",
        y=gemaps_distances["StddevVoicedSegmentLengthSec"],
        x=questions_df["A_perc"],
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "StddevVoicedSegmentLengthSec Feature Correlation Scatter.png",
        ),
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_features_df,
    gemaps_single_f_corr,
    os,
    plot_correlation_heatmap,
):
    plot_correlation_heatmap(
        gemaps_features_df[gemaps_single_f_corr["feature_name"]],
        "Pairwise Pearson Correlation Coefficients (r)",
        save_path=os.path.join(PLOT_SAVE_DIR, "GeMAPS Pairwise Feature Correlations.png"),
        labelsize=12,
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_features_df, os, plot_correlation_bar, track_df):
    plot_correlation_bar(
        title="GeMAPS Feature Correlations (Singers' Gender)",
        feature_df=gemaps_features_df,
        target_feature=track_df["pred_p_male"],
        top_x=10,
        save_path=os.path.join(PLOT_SAVE_DIR, "GeMAPS Feature Correlations (with Singers' Gender).png"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gender dependent
    """)
    return


@app.cell
def _(PLOT_SAVE_DIR, np, os, plt, questions_df):
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    bin_labels = [
        "Only Female Voices",
        "Female Reference Voices, Male Target Voice",
        "Mixed Reference Voices",
        "Male Reference Voices, Female Target Voice",
        "Only Male Voices",
    ]
    counts, bin_edges = np.histogram(questions_df["gender_distribution"], bins=bins)
    plt.figure(figsize=(10, 4), dpi=150)
    bars = plt.barh(bin_labels, counts, color="skyblue", edgecolor="black")
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10,
        )
    plt.xlabel("Number of Questions")
    # plt.ylabel("Gender Distribution Categories")
    plt.title("Gender Distribution Counts in Survey")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_SAVE_DIR, "Counts of Data by Gender Feature Distributions.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    return


@app.cell
def _(questions_df):
    gender_m_mask = questions_df["gender_distribution"] >= 0.75
    gender_f_mask = questions_df["gender_distribution"] <= 0.25
    gender_mixed_mask = questions_df["gender_distribution"] == 0.5
    return gender_f_mask, gender_m_mask, gender_mixed_mask


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_m_mask,
    os,
    plot_correlation_bar,
    questions_df,
):
    gemaps_f_m = plot_correlation_bar(
        title="GeMAPS Feature Correlations (Male Only)",
        feature_df=gemaps_distances[gender_m_mask],
        target_feature=questions_df[gender_m_mask]["A_perc"],
        top_x=10,
        output=True,
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "GeMAPS Feature Correlations (Individual, Male Only).png",
        ),
    )
    gemaps_f_m
    return (gemaps_f_m,)


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_m_mask,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="GeMAPS F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope (Male Only)",
        feature_name="F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        y=gemaps_distances[gender_m_mask]["F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope"],
        x=questions_df[gender_m_mask]["A_perc"],
        save_path=os.path.join(
            PLOT_SAVE_DIR, "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope Feature Correlation Scatter (Male Only).png"
        ),
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_f_mask,
    os,
    plot_correlation_bar,
    questions_df,
):
    gemaps_f_f = plot_correlation_bar(
        title="GeMAPS Feature Correlations (Female Only)",
        feature_df=gemaps_distances[gender_f_mask],
        target_feature=questions_df[gender_f_mask]["A_perc"],
        top_x=10,
        output=True,
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "GeMAPS Feature Correlations (Individual, Female Only).png",
        ),
    )
    gemaps_f_f
    return (gemaps_f_f,)


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_f_mask,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="GeMAPS StddevVoicedSegmentLengthSec (Female Only)",
        feature_name="StddevVoicedSegmentLengthSec",
        y=gemaps_distances[gender_f_mask]["StddevVoicedSegmentLengthSec"],
        x=questions_df[gender_f_mask]["A_perc"],
        save_path=os.path.join(
            PLOT_SAVE_DIR, "StddevVoicedSegmentLengthSec Feature Correlation Scatter (Female Only).png"
        ),
    )
    return


@app.cell
def _(PLOT_SAVE_DIR, gemaps_f_f, gemaps_f_m, os, plot_feature_connection):
    plot_feature_connection(
        set_1=gemaps_f_m["feature_name"],
        set_2=gemaps_f_f["feature_name"],
        set_1_label="Male",
        set_2_label="Female",
        title="Top Feature Correlations Depending on Gender",
        save_path=os.path.join(PLOT_SAVE_DIR, "Top 10 GeMaps Feature Correlations by Gender.png"),
    )
    return


@app.cell
def _(answers_df, gender_mixed_mask, questions_df):
    len(answers_df[answers_df.questionID.isin(questions_df[gender_mixed_mask].index)])
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_distances,
    gender_mixed_mask,
    os,
    plot_correlation_bar,
    questions_df,
):
    gemaps_f_mixed = plot_correlation_bar(
        title="GeMAPS Feature Correlations (Mixed Gender Only)",
        feature_df=gemaps_distances[gender_mixed_mask],
        target_feature=questions_df[gender_mixed_mask]["A_perc"],
        top_x=15,
        output=True,
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "GeMAPS Feature Correlations (Individual, Mixed Gender Only).png",
        ),
    )
    gemaps_f_mixed
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(gemaps_features_df, get_global_distance_scores, questions_df):
    gemaps_gda_df = get_global_distance_scores(gemaps_features_df, questions_df, feature_name="gemaps")
    gemaps_gda_df
    return (gemaps_gda_df,)


@app.cell
def _(PLOT_SAVE_DIR, gemaps_gda_df, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="GeMAPS Features Correlations (All)",
        feature_df=gemaps_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
        save_path=os.path.join(PLOT_SAVE_DIR, "GeMAPS Features Correlations (All).png"),
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    gemaps_gda_df,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="GeMAPS Features (Canberra)",
        feature_name="Features_Canberra",
        y=gemaps_gda_df["gemaps_distance_canberra"],
        x=questions_df["A_perc"],
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "GeMaps Features Correlation Scatter (Canberra Distance).png",
        ),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # ComParE Feature Set
    """)
    return


@app.cell
def _(opensmile):
    smile_compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return (smile_compare,)


@app.cell
def _(DATASET_FOLDER, os):
    compare_feature_path = os.path.join(DATASET_FOLDER, "fma_large_feature_sets", "survey_2_compare.npy")
    return (compare_feature_path,)


@app.cell
def _():
    """
    compare_features = pd.DataFrame(
        track_df.song_path.apply(
            lambda x: get_feature_set(x, smile_compare)
        ).tolist(),
        columns=smile_compare.feature_names,
        index=track_df.index,
    )
    compare_features


    with open(compare_feature_path, "wb") as npyfile:
        np.save(npyfile, compare_features.values)
    """
    return


@app.cell
def _(compare_feature_path, np, pd, scale_df, smile_compare, track_df):
    compare_features_df = pd.DataFrame(
        np.load(compare_feature_path),
        columns=smile_compare.feature_names,
        index=track_df.index,
    )
    compare_features_df = scale_df(compare_features_df)
    compare_features_df
    return (compare_features_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Single Feature Agreement
    """)
    return


@app.cell
def _():
    """
    compare_agreements = get_all_scores(
        compare_features_df, compare_features_df.columns
    )
    top_compare_score_values = get_mean_values(compare_agreements, top_x=15)

    plot_scores(
        x=top_compare_score_values.values(),
        y=top_compare_score_values.keys(),
        title=f"ComParE Single Feature Accuracy (Top {TOP_X})",
        random_chance=RANDOM_CHANCE,
        xlabel="Mean Accuracy (%)",
        save_path=os.path.join(PLOT_SAVE_DIR, "compare_single_accuracy.png"),
    )
    """
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Set Agreement
    """)
    return


@app.cell
def _(compare_features_df, get_global_distance_scores, questions_df):
    compare_gda_df = get_global_distance_scores(compare_features_df, questions_df, feature_name="compare")
    compare_gda_df
    return (compare_gda_df,)


@app.cell
def _(PLOT_SAVE_DIR, compare_gda_df, os, plot_correlation_bar, questions_df):
    plot_correlation_bar(
        title="ComParE Feature Set Correlations",
        feature_df=compare_gda_df,
        target_feature=questions_df["A_perc"],
        top_x=4,
        save_path=os.path.join(PLOT_SAVE_DIR, "ComParE Features Correlations (All).png"),
    )
    return


@app.cell
def _(
    PLOT_SAVE_DIR,
    compare_gda_df,
    os,
    plot_correlation_scatter,
    questions_df,
):
    plot_correlation_scatter(
        title="ComParE Feature Set (Cosine)",
        feature_name="ComParE_Feature_Set_Cosine",
        x=questions_df["A_perc"],
        y=compare_gda_df["compare_distance_cosine"],
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "ComParE Features Correlation Scatter (Cosine Similarity).png",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Voice Quality High Level Features

    Use only cosine distances
    """)
    return


@app.cell
def _(compare_features_df, gemaps_features_df):
    from src.statistics.opensmile_mapping import FEATURE_MAP, convert_to_voice_quality_features

    VOICE_QUALITY_FEATURES = list(FEATURE_MAP.keys())
    voice_quality_df = convert_to_voice_quality_features(gemaps_features_df, compare_features_df)
    voice_quality_df
    return VOICE_QUALITY_FEATURES, voice_quality_df


@app.cell
def _(voice_quality_df):
    voice_quality_df["Shimmer"]
    return


@app.cell
def _(
    VOICE_QUALITY_FEATURES,
    get_distance_row,
    mo,
    pd,
    questions_df,
    voice_quality_df,
):
    vq_distance_diff_df = pd.DataFrame()
    for vq_feature in mo.status.progress_bar(
        VOICE_QUALITY_FEATURES,
        title="Calculating Global Distances",
        remove_on_exit=True,
    ):
        vq_distance_diff_df[vq_feature] = questions_df.apply(
            lambda x: get_distance_row(x, voice_quality_df[vq_feature], "canberra"),
            axis=1,
        )
    vq_distance_diff_df
    return (vq_distance_diff_df,)


@app.cell
def _(
    PLOT_SAVE_DIR,
    os,
    plot_correlation_bar,
    questions_df,
    vq_distance_diff_df,
):
    plot_correlation_bar(
        title="Correlation of Voice Quality Features with Subjective Similarity Ratings",
        feature_df=vq_distance_diff_df,
        target_feature=questions_df["A_perc"],
        top_x=10,
        # output=True
        save_path=os.path.join(
            PLOT_SAVE_DIR,
            "Voice Quality Feature Correlations (Individual).png",
        ),
    )
    return


if __name__ == "__main__":
    app.run()
