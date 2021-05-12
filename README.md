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
## Preprocessing class merge
The associated face with masks detection dataset contains three classes: `with_mask`, `without_mask`, and `mask_weared_incorrect`. The difference between `mask_weared_incorrect` and `with_mask` is visually very minor (e.g., the difference between a mask being worn over nose and not or slightly not). Also, the number of `mask_weared_incorrect` instances is small compared to the other classes (123 instances vs 3232 and 717 for `with_mask` and `without_mask`, respectively). Furthermore, our face mask classifier dataset only contains two classes: faces with masks and faces without masks. Thus, we decided to merge the `mask_weared_incorrect` class into the `with_mask` class. We've written a script that iterates over all of the XML annotation files and makes replacements where appropriate (it also returns the class distributions).

Usage,
```
python src/data/replace_class.py --dir PATH_TO_ANNOTATIONS
```
`PATH_TO_ANNOTATIONS` should contain the original unzipped annotations directory as downloaded from Kaggle. Well, it can be any directory that contains ONLY xml files.

## Automatically reorganize face detection dataset directory structure
<div id="reorganize"></div>
This automatically splits the face detection dataset into train, validation, and testing sets according to user specified proportions and places these files into a directory structure as required by the [ImageAI library](https://github.com/OlafenwaMoses/ImageAI) used for using transfer learning to train a custom YoloV3 object detector. The default split strategy is 5% allocated for the test set with the remaining date being split 80/20 for training/validation.

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
We train classifiers to be used as part of the two-stage object detector. We've used `SGDClassifier`, `RandomForestClassifier`, and transfer learning on `MobileNetV2`. Validation reports are saved to the `results/` directory and each classifier saves its model to the user specified MODEL_OUTPUT directory.

Usage,
```
python train_classifier.py --train TRAINING_SET --val VALIDATION_SET --model_out MODEL_OUTPUT --classifier CLASSIFIER
```
Valid inputs for CLASSIFIER are "MobileNet", "RandomForest", and "LogisticRegression".

## Training custom face detector
For training a custom face mask detector from scratch we use transfer learning using the [ImageAI](https://github.com/OlafenwaMoses/ImageAI). This library currently only provides support for building a custom detector using a pretrained YoloV3 network. There are specific requirements in terms of the structure of the input dataset so it is important to first follow the section <a href="#reorganize">Automatically reorganize face detection dataset directory structure </a> above.

Usage,
```
python train_detector.py --data PATH_TO_REORGANIZED_DATASET
```
Note: this is a very expensive process. Also ensure that the pretrained model is in the `pretrained_models` directory at the root of the project, i.e., `pretrained_models/pretrained-yolov3.h5`. This model should be available in this repo but can also be downloaded from ImageAI [here](https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5).
## Running detect faces as mask vs non-mask
