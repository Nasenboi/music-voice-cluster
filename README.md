# Similarity of Singing Voices

The research repository for my master thesis. The included code was used to create the survey's dataset and to analyze the results. 

## Dataset

The Dataset used in this Project is [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma) \[1\]\[2\], see the repository for more details.

## Structure

The marimo notebooks are ordered by their task and named accordingly. 

### 1. Dataset Preparation

This step is done prior to the survey's publication. The audio, as well as the metadata are formated into a fitting schema, additional features are extracted from the audio and stored in dataframes and finally, the audio dataset is prepared for the ABX-test.

### 2. Statistical Analysis

The users' behavior is analyzed by the answers and user data alone. This analysis includes calculations of agreement measures (percent values and Cohen's Kappa) and two analysis of variance (ANOVAs).

### 3. Feature Comparison

Digital voice representations are compared with the survey results in this step. The hypotheses of the thesis are tested here. The comparison is done by calculating linear correlation coefficients of subjective similarity ratings and algorithmic similarity ratings derived from single or multiple features extracted from the song and vocal stems.

## Run the Notebooks

**Note.** Marimo notebooks do not store the cells' output in contrast to Jupyter notebooks. While this has the advantage of less storage required and better versioning with git, the notebooks have to be run every time to view output. However the results of this thesis are rendered as graphs and plots, grouped by the notebooks name. The code in the notebooks itself should be seen as a reference for what steps were completed to generate the output figures — especially because some notebooks take a long time to complete (for example to run model inference).

#### Requirements
- !! Code tested on for Linux systems so far !! (_should_ work on other systems too though)
- [Python](https://www.python.org/) and [Conda](https://www.anaconda.com/)
- For model inference ideally [CUDA](https://developer.nvidia.com/cuda/toolkit) — the python environments also require CUDA drivers, environments without it have not been tested yet (may cause bugs)
- For infrerence on the included [models](#models) the corresponding weights / model files are required.
- Check if the most important files are present:  

| File | Description |
| ---- | ----------- |
| CSV_FOLDER/large_dataset/dataset_survey_2_final.csv| Contains metadata for the 50 audio tracks |
| DATASET_FOLDER/fma_large_triplets/mel_spec_enc_nlognK_survey_2.npy | Numpy array containing heuristically chosen triplets in form of indicies that are used in the second survey |
| AUDIO_FOLDER/fma_large(_stems)/XYZ | The audio files and vocal stems |


#### Steps

1. Create and activate the conda environments. The base environment _sosv_ is exported to [environment.yml](environment.yml) and used for most of the notebooks. The second environment _sosv-np1_ ([environment-np-1.yml](environment-np-1.yml)) is used to tun inference on the [CVSM model](#contrastive-vocal-similarity-modeling-cvsm), as it requires tensorflow instead of torch and numpy version 1.
```bash
conda create --file environment-npy-1.yml
```
```bash
conda create --file environment.yml
```
```bash
conda activate sosv # or sosv-np1 for notebook 3.3.2
```
2. Create and check the `.env` file. The paths should point to the dataset folders:
    - WORK_PATH: the parent path where the project files are located (optional)
    - DATASET_FOLDER: the folder for the Free Music Archive Dataset and additional CSV files created for — and resulting from the survey
    - MODEL_FOLDER: the folder for machine learning models, each have their own subfolder.
    - CSV_FOLDER: additional CSV files, including checkpoints of the subjective audio labeling and the whole song and metadata dataset — one of the most important files for this repository.
    - PLOT_FOLDER: in here are the plots created by Python notebooks — each with their own subfolder.
    - AUDIO_FOLDER: the folder for the audio files, with the subfolders `fma_large` and `fma_large_stems`
3. There is a function to load the dataset tables used in most of the notebooks: `load_survey_data` from [survey dataset helpers](src/survey_dataset_helpers.py) which takes a dictionary of file paths as input. This is the essential function that needs to work for all subsequent steps. Run this function from any notebook, like [2.1 Examine Survey Results](02_01_examine_survey_results.py) to check if the environment variables and paths are correct.
4. To run a notebook, like [2.1 Examine Survey Results](02_01_examine_survey_results.py) run the command below, this should open a new window in your default browser with the chosen notebook active.
```bash
marimo edit 02_01_examine_survey_results.py
```

## References

\[1\]: Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. 18th International Society for Music Information Retrieval Conference (ISMIR). 18th International Society for Music Information Retrieval Conference (ISMIR). https://arxiv.org/abs/1612.01840  
\[2\]: Defferrard, M., Mohanty, S. P., Carroll, S. F., & Salathé, M. (2018). Learning to Recognize Musical Genre from Audio. The 2018 Web Conference Companion. The 2018 Web Conference Companion. https://doi.org/10.1145/3184558.3192310  

## Models

#### Contrastive Vocal Similarity Modeling (CVSM)

[GitHub](https://github.com/cgaroufis/CVSM)  
Garoufis, C., Zlatintsi, A., & Maragos, P. (2025). CVSM: Contrastive Vocal Similarity Modeling. arXiv [Eess.AS]. [doi:10.48550/arXiv.2510.03025](https://doi.org/10.48550/arXiv.2510.03025)

#### SSL Singer Identity Embedding Model

[GitHub](https://github.com/SonyCSLParis/ssl-singer-identity)  
[HuggingFace](https://huggingface.co/BernardoTorres/singer-identity)  
Torres, B., Lattner, S., & Richard, G. (2023). Singer Identity Representation Learning using Self-Supervised Techniques. International Society for Music Information Retrieval Conference (ISMIR 2023).

#### ECAPA-TDNN Emedding Model

Embedding model from: [GitHub](https://github.com/TaoRuijie/ECAPA-TDNN)  
ECAPA-TDNN Approach: Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification. Interspeech 2020, 3830–3834. [doi:10.48550/arXiv.2005.07143](https://doi.org/10.48550/arXiv.2005.07143)


#### Discogs-EffNet Emgedding Model & Genre Discogs400

[Discogs EffNet](https://essentia.upf.edu/models.html#discogs-effnet)  
[Genre Discogs400](https://essentia.upf.edu/models.html#genre-discogs400)  
[Discogs](https://www.discogs.com/)  
Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2022). Music Representation Learning Based on Editorial Metadata from Discogs. International Society for Music Information Retrieval Conference (ISMIR). [doi:10.48550/arXiv.2309.16418](https://doi.org/10.48550/arXiv.2309.16418)

#### Mel-Band RoFormer for Music Source Separation

[GitHub](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)  
[Hugging Face](https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt)  
Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation. arXiv [Cs.SD]. [doi:10.48550/arXiv.2510.03025](https://doi.org/10.48550/arXiv.2510.03025)

Separator Python class from: [GitHub](https://github.com/nomadkaraoke/python-audio-separator)

#### Voice Gender Classifier 

[GitHub](https://github.com/JaesungHuh/voice-gender-classifier)  
[Hugging Face](https://huggingface.co/JaesungHuh/voice-gender-classifier)  

#### Age Regression Model

[GitHub](https://github.com/griko/voice-age-regression)  
[Hugging Face](https://huggingface.co/griko/age_reg_svr_ecapa_librosa_voxceleb2)  
Koushnir, G., Fire, M., Alpert, G. F., & Kagan, D. (2025). VANPY: Voice Analysis Framework. arXiv [Cs.SD]. [doi:10.48550/arXiv.2502.17579](https://doi.org/10.48550/arXiv.2502.17579)

#### Approachability Regression Model

[Essentia](https://essentia.upf.edu/models.html#approachability)  
Lizarraga, X. (2022). approachability_regression. Essentia.

#### Danceability Classification Model

[Essentia](https://essentia.upf.edu/models.html#danceability)  
Alonso, P. (2022). danceability classifier. Essentia.

#### Engagement Regression Model

[Essentia](https://essentia.upf.edu/models.html#engagement)  
Lizarraga, X. (2022). engagement_regression. Essentia.

#### Mood and Theme Model

[Essentia](https://essentia.upf.edu/models.html#mtg-jamendo-mood-and-theme)  
[GitHub](https://github.com/MTG/mtg-jamendo-dataset)  
Bogdanov, D., Won, M., Tovstogan, P., Porter, A., & Serra, X. (2019). The MTG-Jamendo Dataset for Automatic Music Tagging. Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019). Retrieved from [http://hdl.handle.net/10230/42015](http://hdl.handle.net/10230/42015)

#### TempoCNN

[Essentia](https://essentia.upf.edu/models.html#tempocnn)  
Schreiber, H., & Müller, M. (2019). Musical Tempo and Key Estimation using Convolutional Neural Networks with Directional Filters. Proceedings of the Sound and Music Computing Conference (SMC), 47–54.   
Schreiber, H., & Müller, M. (2018). A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network. International Society for Music Information Retrieval Conference (ISMIR).

#### Qwen-ASR-1.7B Speech Recognition Model 

[GitHub](https://github.com/QwenLM/Qwen3-ASR)  
[Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)  
Qwen3-ASR Technical Report. (2026). arXiv Preprint arXiv:2601. 21337. [doi:10.48550/arXiv.2601.21337](https://doi.org/10.48550/arXiv.2601.21337)

#### Bournemouth Forced Alignment Model

[GitHub](https://github.com/tabahi/bournemouth-forced-aligner)  
[HuggingFace](https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt)  
Rehman, A., Cai, J., Zhang, J.-J., & Yang, X. (2025). BFA: Real-time Multilingual Text-to-speech Forced Alignment. arXiv [doi:10.48550/arXiv.2509.23147](https://doi.org/10.48550/arXiv.2509.23147)  
Rehman, A., Zhang, J.-J., & Yang, X. (2025). CUPE: Contextless Universal Phoneme Encoder for Language-Agnostic Speech Processing. Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP 2025). ICNLSP.