import kaggle
import os

# TODO: Check if these folders are empty, refues to download if not
# TODO: Add second dataset download

faces_dataset_name = "ashishjangra27/face-mask-12k-images-dataset"
# must be called from root dir, find better way to do this
root_raw_data_dir = os.path.join("data", "raw")
kaggle.api.authenticate()

# TODO: Add click default command line arguments as set by make file to defaul dir?
kaggle.api.dataset_download_files(faces_dataset_name,
                                  path=os.path.join(root_raw_data_dir, "face_classifier_data"), 
                                  unzip=True)
# test