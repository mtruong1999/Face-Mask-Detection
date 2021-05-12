import os
from imageai.Detection.Custom import DetectionModelTrainer

PRETRAINED_MODEL = os.path.join("pretrained_models", "pretrained-yolov3.h5")

class YoloV3Custom():

    def __init__(self):
        print("[INFO]: Training custom YoloV3 network")
    
    def train(self, data_dir, classes, batch_size, epochs):
        trainer = DetectionModelTrainer()
        trainer.setModelTypeAsYOLOv3()
        trainer.setDataDirectory(data_directory=data_dir)
        trainer.setTrainConfig(object_names_array=classes, 
                               batch_size=batch_size,
                               num_experiments=epochs,
                               train_from_pretrained_model=PRETRAINED_MODEL)
        trainer.trainModel()
