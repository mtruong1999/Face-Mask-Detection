""" 
This tool is to be used for replacing all "masked_weared_incorrect"
classes to "wear_mask" because our classifier is only binary and does not
support a third class

Also outputs class statistics.

Author: Michael Truong
"""

import xml.etree.ElementTree as ET
import os
import argparse
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="replace class in annotation files")
    parser.add_argument("--old_class",
                        default="mask_weared_incorrect",
                        help="class to be replaced")
    
    parser.add_argument("--new_class",
                       default="with_mask",
                       help="the new replacement class")

    parser.add_argument("--dir",
                       required=True,
                       help="path containing annotations")
    args = parser.parse_args()
    data_dir = args.dir 
    old = args.old_class
    new = args.new_class

    update_count = 0
    class_counts = defaultdict(int)

    for ann in os.listdir(data_dir):
        filename = os.path.join(data_dir, ann)
        tree = ET.parse(filename)
        rootElement = tree.getroot()
        changed = False
        for element in rootElement.findall("object"):
            class_name = element.find("name").text
            class_counts[class_name] += 1

            if class_name == old:
                element.find("name").text = new
                update_count += 1
                changed = True
        if changed:
            tree.write(filename, encoding='UTF-8')
    print("Original class distribution: ")
    for cl, count in class_counts.items():
        print("{} instance count: {}".format(cl, count))
    
    print("{} files were updated.".format(update_count))
