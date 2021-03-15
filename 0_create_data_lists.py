"""
Use this file to unzip and index train and valid files.
This is a simple staff, adapt if needed.
"""

from utils import create_jsons_for_roboflow_pascal_voc


if __name__ == '__main__':
    
    create_jsons_for_roboflow_pascal_voc(
        roboflow_path="trial_dataset",
        output_folder="trial_dataset_dumps",
    )