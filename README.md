# Similarity of Singing Voices

The repository for my master thesis.

## Dataset

The Dataset used in this Project is [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma) \[1\]\[2\], see the repository for more details.

## Structure

The marimo notebooks are ordered by their task and named accordingly. 

### 1. Dataset Preparation

This step is done prior to the surveys' publication. The audio, as well as the metadata are formated into a fitting schema, additional features are extracted from the audio and stored in dataframes and finally, the audio dataset is bein prepared for the ABX-test.

### 2. Statistical Analysis

The survey results are being statistically analized. User behavior is being tracked by the answers and user data alone.

### 3. Feature Comparison

Digital voice representations are being compared with the survey results in this step. The hypotheses of the thesis are being tested here.

## References

\[1\]: Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. 18th International Society for Music Information Retrieval Conference (ISMIR). 18th International Society for Music Information Retrieval Conference (ISMIR). https://arxiv.org/abs/1612.01840  
\[2\]: Defferrard, M., Mohanty, S. P., Carroll, S. F., & Salathé, M. (2018). Learning to Recognize Musical Genre from Audio. The 2018 Web Conference Companion. The 2018 Web Conference Companion. https://doi.org/10.1145/3184558.3192310  

## Models

#### Contrastive Vocal Similarity Modeling (CVSM)

[GitHub](https://github.com/cgaroufis/CVSM)  
Garoufis, C., Zlatintsi, A., & Maragos, P. (2025). CVSM: Contrastive Vocal Similarity Modeling. arXiv [Eess.AS]. [doi:10.48550/arXiv.2510.03025](https://doi.org/10.48550/arXiv.2510.03025)

#### Mel-Band RoFormer for Music Source Separation

[GitHub](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)  
[Hugging Face](https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt)  
Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation. arXiv [Cs.SD]. [doi:10.48550/arXiv.2510.03025](https://doi.org/10.48550/arXiv.2510.03025)

Separator Python class from: [GitHub](https://github.com/nomadkaraoke/python-audio-separator)

#### ECAPA-TDNN Emedding Model

Embedding model from: [GitHub](https://github.com/TaoRuijie/ECAPA-TDNN)  
ECAPA-TDNN Approach: Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification. Interspeech 2020, 3830–3834. [doi:10.48550/arXiv.2005.07143](https://doi.org/10.48550/arXiv.2005.07143)

#### Voice Gender Classifier 

[GitHub](https://github.com/JaesungHuh/voice-gender-classifier)  
[Hugging Face](https://huggingface.co/JaesungHuh/voice-gender-classifier)  

#### Age Regression Model

[GitHub](https://github.com/griko/voice-age-regression)  
[Hugging Face](https://huggingface.co/griko/age_reg_svr_ecapa_librosa_voxceleb2)  
Koushnir, G., Fire, M., Alpert, G. F., & Kagan, D. (2025). VANPY: Voice Analysis Framework. arXiv [Cs.SD]. [doi:10.48550/arXiv.2502.17579](https://doi.org/10.48550/arXiv.2502.17579)

#### Discogs-EffNet Emgedding Model & Genre Discogs400

[Discogs EffNet](https://essentia.upf.edu/models.html#discogs-effnet)  
[Genre Discogs400](https://essentia.upf.edu/models.html#genre-discogs400)  
[Discogs](https://www.discogs.com/)  
Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2022). Music Representation Learning Based on Editorial Metadata from Discogs. International Society for Music Information Retrieval Conference (ISMIR). [doi:10.48550/arXiv.2309.16418](https://doi.org/10.48550/arXiv.2309.16418)



#### Approachability Regression Model

[Essentia](https://essentia.upf.edu/models.html#approachability)  
Lizarraga, X. (2022). approachability_regression. Essentia.




#### Danceability Classification Model

[Essentia](https://essentia.upf.edu/models.html#danceability)  
Alonso, P. (2022). danceability classifier. Essentia.

#### Engagement Regression Model

[Essentia](https://essentia.upf.edu/models.html#engagement)  
Lizarraga, X. (2022). engagement_regression. Essentia.


#### TempoCNN

[Essentia](https://essentia.upf.edu/models.html#tempocnn)  
Schreiber, H., & Müller, M. (2019). Musical Tempo and Key Estimation using Convolutional Neural Networks with Directional Filters. Proceedings of the Sound and Music Computing Conference (SMC), 47–54.   
Schreiber, H., & Müller, M. (2018). A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network. International Society for Music Information Retrieval Conference (ISMIR).