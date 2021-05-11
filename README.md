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

## Automatically reorganize face detection dataset directory structure
This automatically splits the face detection dataset into train, validation, and testing sets according to user specified proportions and places these files into a directory structure as required by the [`ImageAI` library](https://github.com/OlafenwaMoses/ImageAI) used for using transfer learning to train a custom YoloV3 object detector. The default split strategy is 5% allocated for the test set with the remaining date being split 80/20 for training/validation.

Usage,
```
python src/data/split_data.py --images PATH_TO_IMAGES --annot PATH_TO_ANNOTATIONS --dest DESTINATION_DIRECTORY
```
The amount of images to split from the TOTAL image set for training purposes can be custom set using the `--train-size` flag. Remaining data is used for testing. The amount of images to split from the TRAINING portion for validation purposes can be custom set using the `--val-size` flag. These proportions must be in range 0-1.

The resulting file structure should be as follows:
|---DESTINATION_DIRECTORY
    |---train
    |   |---images
    |   |---annotations
    |---test
    |   |---images
    |   |---annotations
    |---validation
    |   |---images
    |   |---annotation

## Training mask/no-mask binary classifier

## Training face detector

## Running detect faces as mask vs non-mask