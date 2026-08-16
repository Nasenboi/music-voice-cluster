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
    import shutil

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    # utils.py file
    # in: FMA: A Dataset For Music Analysis
    # Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR).
    # available under "https://github.com/mdeff/fma"
    from src.submodules.FMA.utils import get_audio_path, load
    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
    )
    from src.survey_dataset_helpers import load_survey_data

    return (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        load_survey_data,
        mo,
        os,
        shutil,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Load the Dataset
    """)
    return


@app.cell
def _(AUDIO_FOLDER, CSV_FOLDER, DATASET_FOLDER, load_survey_data, os):
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
    SURVEY_DATA = load_survey_data(CSV_PATHS)
    questions_df = SURVEY_DATA["questions_df"]
    answers_df = SURVEY_DATA["answers_df"]
    participants_df = SURVEY_DATA["participants_df"]
    songs_df = SURVEY_DATA["songs_df"]
    human_agreement = SURVEY_DATA["human_agreement"]
    answer_a_b_ratio = SURVEY_DATA["answer_a_b_ratio"]
    track_df = SURVEY_DATA["track_df"]
    return (track_df,)


@app.cell
def _(os, track_df):
    missing_audios = []

    def check_file_existence(row):
        if not os.path.exists(row["song_path"]):
            missing_audios.append(row["song_path"])
        if not os.path.exists(row["vocal_path"]):
            missing_audios.append(row["vocal_path"])

    track_df.apply(check_file_existence, axis=1)
    missing_audios
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Move Relevant Files to "Release" Folder
    """)
    return


@app.cell
def _(AUDIO_FOLDER, os, shutil):
    OLD_PATH = AUDIO_FOLDER
    NEW_PATH = AUDIO_FOLDER.replace("SOSV", "Release")

    def moveFiles(row):
        old_song_path = row["song_path"]
        old_vocal_path = row["vocal_path"]
        new_song_path = old_song_path.replace(OLD_PATH, NEW_PATH)
        new_vocal_path = old_vocal_path.replace(OLD_PATH, NEW_PATH)

        os.makedirs(os.path.dirname(new_song_path), exist_ok=True)
        os.makedirs(os.path.dirname(new_vocal_path), exist_ok=True)
        shutil.move(old_song_path, new_song_path)
        shutil.move(old_vocal_path, new_vocal_path)

    return (moveFiles,)


@app.cell
def _(df, moveFiles):
    df.apply(moveFiles, axis=1)
    return


if __name__ == "__main__":
    app.run()
