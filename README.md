# Face Mask Detector
Machine learning app that detects faces of people and classifies whether they are wearing a mask.

## Setup
Create and activate conda environment:
```
conda env create -f environment.yml -n myenv
conda activate myenv
```
## Download dataset
Note, downloading the dataset requires a [Kaggle](www.kaggle.com) account. The dataset can be automatically downloaded from Kaggle by first getting an API token from your Account Tab (`https://www.kaggle.com/<username>/account`), adding the downloaded `kaggle.json` to your `~/.kaggle/` folder (Mac/Linux) or to your `%USERPROFILE%\.kaggle\` folder (or by creating and then running the `download.py` script:
```
python download.py
```
(alternatively, we could have set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` using values from the `kaggle.json`)
