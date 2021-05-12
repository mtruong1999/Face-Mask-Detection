""" 
Trains a user specified classifier on the masked faces dataset.
Model is saved to models/ in project root.

Author: Michael Truong
"""
import argparse
import classifiers
import os


CLASSIFIER = {"LogisticRegression" : classifiers.LogisticRegression,
              "CNN" : classifiers.CNN,
              "MobileNet" : classifiers.MobileNet,
              "RandomForest" : classifiers.RandomForest,
              "ResNet" : classifiers.ResNet}

def get_classes(path):
    return [c for c in os.listdir(path) if os.path.isdir(c)]

def check_directory(path):
    if not os.path.isdir(path):
        raise Exception("{} does not exist".format(path))

def get_classifier(classifier_name):
    return CLASSIFIER[classifier_name]

def train_model(train_path, val_path, model_path, 
                img_width, img_height, augment, 
                save_history, classifier_name):
    
    model = get_classifier(classifier_name)(train_path, val_path, img_width, img_height, augment, save_history)
    model.train(model_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a masked face classifier")

    parser.add_argument("--classifier",
                        default="MobileNet",
                        type=str,
                        help="Classifier model to train, select one of: " + 
                        "LogisticRegression, CNN, MobileNet, RandomForest, ResNet")
    
    parser.add_argument("--train",
                        required=True,
                        type=str,
                        help="Path to classifier training set")
    
    parser.add_argument("--val",
                        required=True,
                        type=str,
                        help="Path to classifier validation set")
    
    parser.add_argument("--model_out",
                        default=None,
                        help="Output path to which to save training models")
    
    args = parser.parse_args()
    train_dir = args.train
    val_dir = args.val
    classifier_choice = args.classifier

    check_directory(train_dir)
    check_directory(val_dir)

    

    labels = get_classes(train_dir)
    # ensure that both directories have same classes
    train_labels = get_classes(val_dir)
    assert set(labels) == set(train_labels), "Inconsistent labels in train and validation sets"

    if classifier_choice not in CLASSIFIER:
        raise Exception("{} is not a valid classifier choice, must be one of {}".format(classifier_choice, CLASSIFIER.keys()))

    model_dir = os.path.join(args.model_out, classifier_choice)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    train_model(train_dir, val_dir, model_dir, 224, 224, augment=False, save_history=True, classifier_name=classifier_choice)
    


