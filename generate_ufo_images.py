import cv2
import numpy as np
import tqdm

def bicubic_downsample(image, scale=2):
    # Resize the image using bicubic interpolation
    downsampled_image = cv2.resize(image, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC)

    return downsampled_image

import os

# Define the input and output directories
input_dir = '../DataSet/UFO-120/train_val/lr_2x'

  
output_dir_3x = '../DataSet/UFO-120/train_val/lr_3x'
output_dir_4x = '../DataSet/UFO-120/train_val/lr_4x'
output_dir_8x = '../DataSet/UFO-120/train_val/lr_8x'

# Create the output directory if it doesn't exist
os.makedirs(output_dir_3x, exist_ok=True)
os.makedirs(output_dir_4x, exist_ok=True)
os.makedirs(output_dir_8x, exist_ok=True)

# Get a list of all the image files in the input directory
image_files = os.listdir(input_dir)

# Process each image file
for image_file in image_files:
    # Load the image
    image = cv2.imread(os.path.join(input_dir, image_file))

    # Downsample the image using bicubic interpolation
    downsampled_image_3x = bicubic_downsample(image, scale=3/2)
    downsampled_image_4x = bicubic_downsample(image, scale=2)
    downsampled_image_8x = bicubic_downsample(image, scale=4)

    # Save the downsampled image to the output directory
    output_path_3x = os.path.join(output_dir_3x, image_file)
    output_path_4x = os.path.join(output_dir_4x, image_file)
    output_path_8x = os.path.join(output_dir_8x, image_file)
    cv2.imwrite(output_path_3x, downsampled_image_3x)
    cv2.imwrite(output_path_4x, downsampled_image_4x)
    cv2.imwrite(output_path_8x, downsampled_image_8x)