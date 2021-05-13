import argparse
import os
import sys
#from imageai.Detection import Custom
import cv2
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
#from third_party.imageai.Detection.Custom import CustomObjectDetection

#from third_party.imageai.Detection.Custom import DetectionModelTrainer

ARCHITECTURES = ["yolov3"]
PICKLE_MODELS = ["RandomForest", "SGD"]
IOU_THRESH = 0.5
OBJECT_THRESH = 0.3
NMS_THRESH = 0.5

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
    img = cv2.imread("data\\friends.jpg")

    model = CustomObjectDetection()
    model.setModelTypeAsYOLOv3()
    model.setModelPath(model_path)
    model.setJsonPath(os.path.join(json_path, "detection_config.json"))
    model.loadModel()
    detected_image, detections = model.detectObjectsFromImage(input_image=img, input_type="array", output_type="array", minimum_percentage_probability=30)

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
    
    cv2.imshow("Main Image", detected_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces on an image or dataset")
    
    parser.add_argument("--detector",
                        type=str,
                        default=None,
                        help="CNN architecture to use for detector")
    
    parser.add_argument("--classifier_model",
                        type=str,
                        help="Path to model for classifying faces")

    parser.add_argument("--detection_model",
                        type=str,
                        help="Path to detection model")
    
    parser.add_argument("--data_path",
                        type=str,
                        default=None,
                        help="Only relevant for YoloV3, path to data directory")
    
    parser.add_argument("--evaluate",
                        type=int,
                        default=False,
                        help="Only relevant for YoloV3, flag for evaluating training models")

    args = parser.parse_args()
    detector = args.detector
    detection_model = args.detection_model
    classifier_model = args.classifier_model
    data_path = args.data_path
    yolo_evaluate = bool(args.evaluate)

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
    

    
