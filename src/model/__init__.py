'''
This package contains all of the models for classifying
mask vs no-mask. They should all have the same interface 
and will be used for training on a dataset and saving the 
trained model to models/ in the project root.
'''
from .CNN import CNN
from .LogisticRegression import LogisticRegression
from .MobileNet import MobileNet
from .RandomForest import RandomForest
from .ResNet import ResNet
