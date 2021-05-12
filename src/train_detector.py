import argparse
import os
import sys
import detectors

ARCHITECTURES = {"yolov3" : detectors.YoloV3Custom}
# object classes as found in image xml files
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

def check_directory_structure(data_dir):
    if not os.path.isdir(data_dir):
        raise Exception("[ERROR]: Directory \"" + data_dir + "\" does not exist.")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise Exception("[ERROR]: Missing train and validation folders")

    if not os.path.isdir(os.path.join(train_dir, "images")) or not os.path.isdir(os.path.join(train_dir, "annotations")):
        raise Exception("[ERROR]: Missing images/annotations folders inside of {}".format(train_dir))
    
    if not os.path.isdir(os.path.join(val_dir, "images")) or not os.path.isdir(os.path.join(val_dir, "annotations")):
        raise Exception("[ERROR]: Missing images/annotations folders inside of {}".format(val_dir))

def get_architecture(architecture):
    return ARCHITECTURES[architecture]()

def train_and_save_model(architecture, data_path, classes, batch_size, epochs):
    model = get_architecture(architecture)
    model.train(data_path, classes, batch_size, epochs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a masked-face detector")

    parser.add_argument("--data", "-d",
                        required=True,
                        type=str,
                        help="path to directory containing \
                        'train' & 'validation' subfolders")
    
    parser.add_argument("--algorithm", "-a",
                        type=str,
                        default="yolov3",
                        help="CNN architecture to use for detector")
    
    args = parser.parse_args()
    algorithm = args.algorithm
    data_dir = args.data

    check_directory_structure(data_dir)

    if algorithm not in ARCHITECTURES:
        print("[ERROR]: Invalid algorithm choice passed, must be one of "+
              str(list(ARCHITECTURES.keys())) + ".")
        sys.exit(0)

    train_and_save_model(algorithm, data_dir, CLASSES, batch_size=4, epochs=100)
