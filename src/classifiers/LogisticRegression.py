import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import os
import seaborn as sn
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread
from skimage.transform import resize
RESULTSPATH = "results"
LOSS = "log"

class LogisticRegression():

    def __init__(self, train_path, val_path, width, height, augment=False, save_history=True):
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.img_width, self.img_height = width, height
        self.save_history = save_history
    
    def train(self, model_output):
        sgd_clf = SGDClassifier(alpha=0.0001, random_state=0, loss=LOSS) # log performs logistic regression
        X_train, Y_train, X_test, Y_test = self._get_data()
        print("Training now....")
        sgd_clf.fit(X_train, Y_train)
        print("Training complete...")

        print("Predicting on validation...")
        Y_predicted = sgd_clf.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predicted)
        print("Validation accuracy: {:.3f}".format(accuracy))

        print("Generating reports and saving reports to {}/...".format(RESULTSPATH))
        report = classification_report(Y_test, Y_predicted, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(RESULTSPATH,"SGDClassifier_validation_report.csv"))

        print("Generating and saving confusion matrix...")
        confusionMatrix = confusion_matrix(Y_test, Y_predicted)

        plt.style.use("dark_background")
        plt.figure(figsize=(9,8))
        sn.heatmap(confusionMatrix, annot=True, cmap="Blues", fmt="d")
        plt.title("SGDClassifier confusion matrix")
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.savefig(os.path.join(RESULTSPATH, "SGDClassifier_confusionmatrix.png"))

        if model_output:
            filename = os.path.join(model_output, "SGDClassifier_loss-{}.pkl".format(LOSS))
            pickle.dump(sgd_clf, open(filename, 'wb'))
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

    def _batch(self, X, Y, n=1):
        """Generator for data to avoid holding all images in memory

        Args:
            X (iterable): Data features
            Y (iterable): Data labels
            n (int, optional): Batch size. Defaults to 1.

        Yields:
            tuple: Batch size amount of data and labels
        """
        length = len(X)
        for i in range(0, length, n):
            yield X[i : min(i + n, length)], Y[i : min(i + n, length)]