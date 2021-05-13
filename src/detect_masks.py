import argparse
import os
import sys
from imutils import paths
import pickle
import cv2
import numpy as np

from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
#from third_party.imageai.Detection.Custom import CustomObjectDetection

#from third_party.imageai.Detection.Custom import DetectionModelTrainer

ARCHITECTURES = ["yolov3", "ssd"]
PICKLE_MODELS = ["RandomForest", "SGD"]
IOU_THRESH = 0.5
OBJECT_THRESH = 0.3
NMS_THRESH = 0.5
YOLOv3_OUTPUT_PREDICTIONS = os.path.join('data', 'yolo_predictions') # TODO: auto create these directories
SSD_OUTPUT_PREDICTIONS = os.path.join('data', 'ssd_predictions')
SSD_CONFIDENCE_THRESHOLD = 0.5



def evaluate_yolov3(data_path, models_path, json_path, iou_thresh, object_thresh, nms_thresh):
    model = DetectionModelTrainer()
    model.setModelTypeAsYOLOv3()
    model.setDataDirectory(data_directory=data_path)
    model.evaluateModel(model_path=models_path,
                        json_path=os.path.join(json_path, "detection_config.json"),
                        iou_threshold=iou_thresh,
                        object_threshold=object_thresh,
                        nms_threshold=nms_thresh)

def predict_yolov3(data_path, model_path, json_path):
    # source: https://github.com/OlafenwaMoses/ImageAI/blob/master/examples/custom_detection_array_input_output.py
    #img = cv2.imread("data\\two_class_face_detection\\test\\images\\maksssksksss327.png")
    #img = cv2.imread("data\\friends.jpg")
    predictions = {} # dictionary mapping image name to a list of all objects detected (each object is a dictionary)
    model = CustomObjectDetection()
    model.setModelTypeAsYOLOv3()
    model.setModelPath(model_path)
    model.setJsonPath(os.path.join(json_path, "detection_config.json"))
    model.loadModel()

    test_images_path = os.path.join(data_path, "test", "images")
    if not os.path.isdir(test_images_path):
        raise Exception("[Error]: Directory {} does not exist".format(test_images_path))
    
    for path in paths.list_images(test_images_path):
        img = cv2.imread(path)
        # get image name with extension removed
        img_name = os.path.splitext(path.split(os.path.sep)[-1])[0]

        #detected_image, detections = model.detectObjectsFromImage(input_image=img, input_type="array", output_type="array", minimum_percentage_probability=30)
        detections = model.detectObjectsFromImage(input_image=img, input_type="array",
                                                  output_image_path=os.path.join(YOLOv3_OUTPUT_PREDICTIONS, img_name + "_detected.png"),
                                                  minimum_percentage_probability=30)
        
        predictions[img_name] = detections
        #for eachObject in detections:
        #    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        
        #cv2.imshow("Main Image", detected_image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces on an image or dataset")
    
    parser.add_argument("--detector",
                        type=str,
                        default=None,
                        help="CNN architecture to use for detector, must be one of 'yolov3' or 'ssd'")
    
    parser.add_argument("--classifier_model",
                        type=str,
                        help="Path to model for classifying faces")
    # pass OpenCV DNN face detector
    parser.add_argument("--detection_model",
                        type=str,
                        help="Path to detection model")
    parser.add_argument("--proto",
                        type=str,
                        help="Path to proto file if using SSD")
    # For yolov3
    parser.add_argument("--data_path",
                        type=str,
                        default=None,
                        help="Only relevant for YoloV3, path to data directory")
    # For yolov3
    parser.add_argument("--evaluate",
                        type=int,
                        default=False,
                        help="Only relevant for YoloV3, flag for evaluating training models")

    args = parser.parse_args() 

    detector = args.detector # yolov3 or ssd
    detection_model = args.detection_model # path to caffemodel
    proto_path = args.proto # path to prototxt for caffemodel
    classifier_model = args.classifier_model # classifier to use
    data_path = args.data_path # path to dataset
    yolo_evaluate = bool(args.evaluate) 

    if detector not in ARCHITECTURES:
        raise Exception("[ERROR]: Invalid detection algorithm choice passed, must be one of {}".format(ARCHITECTURES))
    if detector == "YoloV3":
        yolov3_models_dir = os.path.join(data_path, "models")
        yolov3_json_dir = os.path.join(data_path, "json")
        if not os.path.isdir(yolov3_models_dir) or not os.path.isdir(yolov3_json_dir):
            raise Exception("[ERROR]: Data path passed for YoloV3 is invalid \
                            or does not have required info")

        if yolo_evaluate:
            print("HERE")
            evaluate_yolov3(data_path, yolov3_models_dir, yolov3_json_dir,
                            iou_thresh=IOU_THRESH,
                            object_thresh=OBJECT_THRESH,
                            nms_thresh=NMS_THRESH)
        else:
            predict_yolov3(data_path, detection_model, yolov3_json_dir)
    
    if classifier_model.endswith(".pkl"):
        print("Model is pickle file, loading now")
        model = pickle.load(open(classifier_model), 'rb')
    else:
        # Source: https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py
        print("Loading detector")
        net = cv2.dnn.readNet(proto_path, detection_model)
        
        print("Loading classifier")
        model = load_model(classifier_model)

        test_images_path = os.path.join(data_path, "test", "images")
        if not os.path.isdir(test_images_path):
            raise Exception("[Error]: Directory {} does not exist".format(test_images_path))

        for path in paths.list_images(test_images_path):
            img = cv2.imread(path)
            (h, w) = img.shape[:2]
            
            # input to network must be 300x300
            net.setInput(cv2.dnn.blobFromImage(img, 1.0, (300,300), (104.0, 177.0, 123.0)))
            
            # get detected objects
            detections = net.forward()

            # classify on all predictions
            for i in range(0, detections.shape[2]):
                # get detection confidence/probability 
                confidence = detections[0, 0, i, 2]

                # check threshold met (yolov3 had this thresh set to 0.3, but we default 0.5)
                if confidence > SSD_CONFIDENCE_THRESHOLD:
                    # get bbox coordinates
                    startX = int(detections[0, 0, i, 3] * w)
                    startY = int(detections[0, 0, i, 4] * h)
                    endX = int(detections[0, 0, i, 5] * w)
                    endY = int(detections[0, 0, i, 6] * h)

                    # check bbox bounds
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w-1, endX), min(h-1, endY))

                    # crop out face and classify
                    cropped_face = img[startY : endY, startX : endX]

                    # resize to network requirement
                    cropped_face = cv2.resize(cropped_face, (224, 224))
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    cropped_face = img_to_array(cropped_face)
                    
                    # normalize image as required by mobilenet
                    cropped_face = preprocess_input(cropped_face)
                    
                    # predict using model
                    print("Predicting....")
                    result = model.predict(np.expand_dims(cropped_face, axis=0))
                    result = float(result)
                    # result is a probability, we use 0.5 as cutoff
                    # if result < 0.5 we classify as with_mask
                    prediction = "WithMask" if result < 0.5 else "WithoutMask"
                    # blue for mask, red for no mask
                    color = (255,0,0) if prediction=="WithMask" else (0,0,255)

                    caption = "{} : {:.3f}".format(prediction, result * 100)

                    # display
                    cv2.putText(img, caption, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                    cv2.rectangle(img, (startX, startY), (endX, endY), color=color, thickness=2)
            
            # write image to output
            img_name = os.path.splitext(path.split(os.path.sep)[-1])[0]
            cv2.imwrite(os.path.join(SSD_OUTPUT_PREDICTIONS, img_name + "_detected.png"), img)

        print("Predictions on test set done, output detections saved to {}".format(SSD_OUTPUT_PREDICTIONS))





