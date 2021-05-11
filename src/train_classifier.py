""" 
Trains a user specified classifier on the masked faces dataset.
Model is saved to models/ in project root.

Author: Michael Truong
"""
from optparse import OptionParser
import model


classifier = {"LogisticRegression" : model.LogisticRegression,
              "CNN" : model.CNN,
              "MobileNet" : model.MobileNet,
              "RandomForest" : model.RandomForest,
              "ResNet" : model.ResNet}

def get_classifier(classifier_name):
    return classifier[classifier_name]

def train_model(data, classifier_name):
    model = get_classifier(classifier_name)


if __name__ == "__main__":
    pass
