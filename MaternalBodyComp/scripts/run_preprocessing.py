import yaml
from pathlib import Path
from maternal_ultrasound.preprocessing import process_non_abd, process_abd, crop_contact_images

# load config
with open("configs/paths.yaml") as f:
    config = yaml.safe_load(f)

# run pipeline
process_non_abd(config["input_folder"], config["output_folder"])
process_abd(config["input_folder"], config["output_folder"], config["reference_image1"], config["reference_image2"])
crop_contact_images(config["put_folder"])