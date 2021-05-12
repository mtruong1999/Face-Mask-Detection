import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import os
import seaborn as sn
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier

RESULTSPATH = "results"

class RandomForest():

    def __init__(self, train_path, val_path, width, height, augment=False, save_history=True):
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.img_width, self.img_height = width, height
        self.save_history = save_history
    
    def train(self, model_output):
        forest_clf = RandomForestClassifier(random_state=0) # log performs logistic regression
        X_train, Y_train, X_test, Y_test = self._get_data()
        print("Training now....")
        forest_clf.fit(X_train, Y_train)
        print("Training complete...")

        print("Predicting on validation...")
        Y_predicted = forest_clf.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predicted)
        print("Validation accuracy: {:.3f}".format(accuracy))

        print("Generating reports and saving reports to {}/...".format(RESULTSPATH))
        report = classification_report(Y_test, Y_predicted, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(RESULTSPATH,"RandomForestClassifier_validation_report.csv"))

        print("Generating and saving confusion matrix...")
        confusionMatrix = confusion_matrix(Y_test, Y_predicted)

        plt.style.use("dark_background")
        plt.figure(figsize=(9,8))
        sn.heatmap(confusionMatrix, annot=True, cmap="Blues", fmt="d")
        plt.title("RandomForestClassifier confusion matrix")
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.savefig(os.path.join(RESULTSPATH, "RandomForestClassifier_confusionmatrix.png"))

        if model_output:
            filename = os.path.join(model_output, "RandomForestClassifier_model.pkl")
            pickle.dump(forest_clf, open(filename, 'wb'))
            # load it later via
            # model = pickle.load(open(filename), 'rb'))

    def _load_images(self, data_path):
        X = []
        Y = []
        for path in paths.list_images(data_path):
            label = path.split(os.path.sep)[-2]
            img = imread(path)
            img = resize(img, (self.img_height, self.img_width))
            img = img.astype(np.float32)
            # normalize image
            img = img.flatten()
            img = img/255.0
            X.append(img)
            Y.append(label)
        return np.array(X, dtype=np.float32), np.array(Y)

    def _get_data(self):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        X_train, Y_train = self._load_images(self.train_path)
        X_test, Y_test = self._load_images(self.val_path)
        
        return X_train, Y_train, X_test, Y_test
