import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    import pathlib

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import langcodes

    from src.phoneme_extractor.helpers import plot_mel_phonemes
    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        MODEL_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
        PLOT_FOLDER,
    )
    from src.survey_dataset_helpers import load_tracks_df
    from src.utils import get_trimmed_audio

    return (
        AUDIO_FOLDER,
        CSV_FOLDER,
        MODEL_FOLDER,
        PLOT_FOLDER,
        get_trimmed_audio,
        langcodes,
        librosa,
        load_tracks_df,
        mo,
        np,
        os,
        pathlib,
        plot_mel_phonemes,
        plt,
        torch,
    )


@app.cell
def _():
    # load after other imports to avoid crashes....
    return


@app.cell
def _():
    from qwen_asr import Qwen3ASRModel

    return (Qwen3ASRModel,)


@app.cell
def _():
    from bournemouth_aligner import PhonemeTimestampAligner

    return (PhonemeTimestampAligner,)


@app.cell
def _(PLOT_FOLDER, os):
    PLOT_SAVE_DIR = os.path.join(PLOT_FOLDER, "survey_2", "01_02_04")
    return (PLOT_SAVE_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Dataset
    """)
    return


@app.cell
def _(AUDIO_FOLDER, CSV_FOLDER, load_tracks_df, os):
    track_df = load_tracks_df(
        {
            "song_files": os.path.join(AUDIO_FOLDER, "fma_large"),
            "stem_files": os.path.join(AUDIO_FOLDER, "fma_large_stems"),
            "tracks": os.path.join(
                CSV_FOLDER,
                "LargeDataset",
                "dataset_survey_2_final.csv",
            ),
        }
    )
    track_df
    return (track_df,)


@app.cell
def _(torch, track_df):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE = 16_000
    SAMPLE_TRACK = track_df.sample(n=1).iloc[0]
    SONG_PATH = SAMPLE_TRACK.song_path
    VOCAL_PATH = SAMPLE_TRACK.vocal_path
    VOCAL_PATH
    return DEVICE, SAMPLE_RATE, VOCAL_PATH


@app.cell
def _(VOCAL_PATH, pathlib):
    TXT_PATH = pathlib.Path(VOCAL_PATH).with_suffix(".txt")
    return


@app.cell
def _(SAMPLE_RATE, VOCAL_PATH, get_trimmed_audio, np):
    # y, sr = librosa.load(VOCAL_PATH, sr=SAMPLE_RATE, mono=True)
    y_snippets = get_trimmed_audio(
        VOCAL_PATH,
        sr=SAMPLE_RATE,
        to_tensor=False,
        concat=False,
        min_duration=2,
    )
    y = np.concatenate(y_snippets, axis=-1)
    return y, y_snippets


@app.cell
def _(SAMPLE_RATE, mo, y_snippets):
    mo.audio(y_snippets[-1], SAMPLE_RATE)
    return


@app.cell
def _(SAMPLE_RATE, librosa, np, plt, y):
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=128, fmax=8000
    )

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=SAMPLE_RATE, fmax=8000, ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automatic Speech Recognition

    - Python Package from [GitHub](https://github.com/QwenLM/Qwen3-ASR)
    """)
    return


@app.cell
def _(MODEL_FOLDER, os):
    asr_model_path = os.path.join(MODEL_FOLDER, "Qwen-ASR-1.7B")
    return (asr_model_path,)


@app.cell
def _(Qwen3ASRModel, asr_model_path, torch):
    asr_model = Qwen3ASRModel.from_pretrained(
        asr_model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    return (asr_model,)


@app.cell
def _():
    # unload
    """
    del asr_model
    # asr_model = None
    torch.cuda.empty_cache()
    """
    return


@app.cell
def _(SAMPLE_RATE, asr_model, y_snippets):
    """
    if not os.path.exists(TXT_PATH):
        asr_result = asr_model.transcribe(audio=(y, SAMPLE_RATE))[0]
        asr_text = asr_result.text
        with open(TXT_PATH, "w") as f:
            f.write(asr_result.text)
    else:
        print("Path exists")
        with open(TXT_PATH, "r") as f:
            asr_text = f.read()
    """

    asr_results = [
        asr_model.transcribe(audio=(s, SAMPLE_RATE))[0] for s in y_snippets
    ]
    asr_texts = [asr.text for asr in asr_results]
    asr_texts[0]
    return asr_results, asr_texts


@app.cell
def _(SAMPLE_RATE, mo, y_snippets):
    mo.audio(y_snippets[0], rate=SAMPLE_RATE)
    return


@app.cell
def _(asr_result, asr_results, langcodes):
    def to_language_code(lang: str) -> str:
        code = langcodes.find(lang)
        return f"{code.language}"

    try:
        asr_lanugage = (
            to_language_code(asr_results[0].language)
            if asr_result.language is not None
            else to_language_code("english")
        )
    except Exception as e:
        asr_lanugage = to_language_code("english")
    asr_lanugage
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forced Speech Alignment for Phenome Snippets

    - Python Package from [GitHub](https://github.com/tabahi/bournemouth-forced-aligner)

    Rehman, A., Cai, J., Zhang, J.-J., & Yang, X. (2025). BFA: Real-time Multilingual Text-to-speech Forced Alignment. https://arxiv.org/abs/2509.23147
    """)
    return


@app.cell
def _(MODEL_FOLDER, os):
    fa_model_path = os.path.join(
        MODEL_FOLDER,
        "bournemouth",
        "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt",
    )
    return (fa_model_path,)


@app.cell
def _(PhonemeTimestampAligner, fa_model_path):
    aligner = PhonemeTimestampAligner(
        preset="asr_lanugage", cupe_ckpt_path=fa_model_path
    )
    return (aligner,)


@app.cell
def _(DEVICE, torch, y_snippets):
    y_tensors = [torch.tensor(s).to(DEVICE) for s in y_snippets]
    wav_tensors = [ten.unsqueeze(0).expand(2, -1) for ten in y_tensors]
    return (wav_tensors,)


@app.cell
def _(SAMPLE_RATE, aligner, wav_tensors):
    audios = [aligner.load_audio(ten, sr=SAMPLE_RATE) for ten in wav_tensors]
    return (audios,)


@app.cell
def _(aligner, asr_texts, audios):
    fa_restult = aligner.process_sentences_batch(asr_texts, audios)
    return (fa_restult,)


@app.cell
def _(PLOT_SAVE_DIR, aligner, audios, fa_restult, os, plot_mel_phonemes):
    mel_spec = aligner.extract_mel_spectrum(
        audios[0].cpu()[0].unsqueeze(0),
        wav_sample_rate=aligner.resampler_sample_rate,
    )

    # --- Phoneme → frame mapping ---
    seg = fa_restult[0]["segments"][0]
    segment_duration = seg["end"] - seg["start"]  # in seconds
    total_frames = mel_spec.shape[0]
    frames_per_second = total_frames / segment_duration

    frames_assorted = aligner.framewise_assortment(
        aligned_ts=seg["phoneme_ts"],
        total_frames=total_frames,
        frames_per_second=frames_per_second,
        gap_contraction=0,
        select_key="phoneme_id",
    )

    frames_assorted = [
        aligner.phoneme_id_to_label.get(pid, "...") for pid in frames_assorted
    ]

    compress_framesed = aligner.compress_frames(frames_assorted)

    plot_mel_phonemes(
        mel_spec,
        compress_framesed,
        save_path=os.path.join(PLOT_SAVE_DIR, "phoneme_split_result.png"),
    )
    return


@app.cell
def _(SAMPLE_RATE, mo, y_snippets):
    mo.audio(y_snippets[0], rate=SAMPLE_RATE)
    return


if __name__ == "__main__":
    app.run()
