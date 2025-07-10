# this is for notebook
import torch
import os
import shutil
from roboflow import Roboflow

#sigm up at https://www.roboflow.com/ and go to https://app.roboflow.com/alexeyvoronin/settings/usage for keys
robo_flow_public_api_key = '*****'
robo_flow_private_api_key = '******'


torch.cuda.set_per_process_memory_fraction(0.4, device=0)

for version in range(9, 10):
    try:
        # Download using Roboflow (default location)
        rf = Roboflow(api_key=robo_flow_private_api_key)
        project = rf.workspace("alexeyvoronin").project("railroad-cars-pyjpo")
        dataset = project.version(version).download("yolov8")

        # Define your desired custom directory
        target_path = "/mnt/c/Kaggle/train_data/rail_cars/"
        os.makedirs(target_path, exist_ok=True)

        # Move contents to custom path
        shutil.move(dataset.location, target_path)

        # Path to data.yaml in new location
        data_yaml_path = os.path.join(target_path, f"data{version}.yaml")
    except Exception as e:
        print(e)
        continue