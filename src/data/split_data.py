""" 
Splits the face detection data into train, validation, testing splits into
a directory structure as required by ImageAI

Author: Michael Truong
"""
import argparse
import os
import sys
import numpy as np

np.random.seed(1)
# supported image types
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tiff")

def train_val_test_split(images_path, train_size, val_size):
    """Splits data set in specified directory into train/test/validate
    sets.

    Args:
        images_path (str): Orignial images directory
        train_size (float): Fraction of FULL data for use in training
        val_size (float): Fraction of TRAINING data for use as validation

    Returns:
        dict: The resulting train/test/validate steps
    """
    images = [f for f in os.listdir(images_path) if f.lower().endswith(IMAGE_TYPES)]
    if not images:
        sys.exit("{} does not contain any images".format(images_path))

    file_count = len(images)
    train_count = round(train_size * file_count)
    val_count = round(val_size * train_count)
    train_count = train_count - val_count
    
    np.random.shuffle(images)

    val_offset = train_count + val_count
    splits = {"train" : images[ : train_count],
           "validation" : images[train_count : val_offset],
           "test": images[val_offset : ]}
    
    return splits

def reorganize_files(images_path, annot_path, dst, splits):
    """Moves images into new directory structure according to 
    whether it belongs to train/test/validation set

    Args:
        images_path (str): Original images directory
        annot_path (str): Original annotations directory
        dst (str): Destination directory to place data organization
        splits (dict): Image names split into train/test/validate sets
    """
    
    for split in splits:
        # if new location for images in split doesnt exist, create it
        new_images_dir = os.path.join(dst, split, "images")
        if not os.path.exists(new_images_dir):
            os.makedirs(new_images_dir)
        
        new_annotations_dir = os.path.join(dst, split, "annotations")
        if not os.path.exists(new_annotations_dir):
            os.makedirs(new_annotations_dir)

        for file in splits[split]:
            # get filename without extension
            base_file_name = os.path.splitext(file)[0]

            # get corresponding annotations filename
            xml_file_name = base_file_name + ".xml"

            if not os.path.exists(os.path.join(annot_path, xml_file_name)):
                print("[WARNING]: annotation file for {} not found, skipping this image".format(file))
            else:
                # move image file to new directory
                os.rename(os.path.join(images_path, file),
                          os.path.join(new_images_dir, file))
                # move annotation file to new directory
                os.rename(os.path.join(annot_path, xml_file_name),
                          os.path.join(new_annotations_dir, xml_file_name))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a face detection data directory " +
                                    "into training, validation, and testing sets. " +
                                    "Images and their corresponding annotations should " +
                                    "have the same name.")

    parser.add_argument("--images",
                        required=True,
                        help="Path to images of dataset")
    
    parser.add_argument("--annot",
                        required=True,
                        help="Path to annotations")

    parser.add_argument("--dest",
                        default=".",
                        help="Path to place output directories")

    parser.add_argument("--train_size",
                        default=0.95,
                        help="Portion of TOTAL data to allocate for training. " + 
                        "Remaining data is allocated for testing. Range: 0-1")
    
    parser.add_argument("--val_size",
                        default=0.2,
                        help="Portion of TRAINING data to allocate for validation. "+
                        "Remaining data is kept as training set. Range: 0-1")

    args = parser.parse_args()

    images_path = args.images
    annot_path = args.annot
    dest_path = args.dest

    if not os.path.exists(images_path):
        sys.exit("[ERROR]: {} does not exist".format(images_path))
    
    if not os.path.exists(annot_path):
        sys.exit("[ERROR]: {} does not exist".format(annot_path))

    train_size = float(args.train_size)
    val_size = float(args.val_size)

    if train_size < 0 or train_size > 1 or val_size < 0 or val_size > 1:
        sys.exit("Invalid train/val size, must be in range (0, 1)")

    splits = train_val_test_split(images_path, train_size, val_size)

    reorganize_files(images_path, annot_path, dest_path, splits)
