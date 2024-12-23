import sys
sys.path.append("") 

import os
import base64
import cv2
import pandas as pd
import numpy as np

def images_to_base64_csv(input_folder, output_csv):
    """
    Read all images inside the input folder (including subfolders),
    convert them to base64 along with their paths, and save them to a CSV file.

    Args:
        input_folder (str): Path to the root folder containing images.
        output_csv (str): Path to save the output CSV file.
    """
    image_data = []

    # Walk through all files in the directory and subdirectories
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                # Read the image and encode it to base64
                with open(file_path, "rb") as image_file:
                    base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')

                image_data.append({
                    "image_path": file_path,
                    "base64": base64_encoded
                })
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Convert list of dicts to a pandas DataFrame
    df = pd.DataFrame(image_data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

import pandas as pd
import base64
import cv2
import numpy as np
import os

def base64_to_image(csv_file, output_folder):
    """
    Read the first row of the CSV, decode the base64 image, and save it to the output folder.

    Args:
        csv_file (str): Path to the CSV file containing image data.
        output_folder (str): Path to save the output image.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    if df.empty:
        print("The CSV file is empty.")
        return

    # Get the first row
    first_row = df.iloc[0]

    image_path = first_row['image_path']
    base64_encoded = first_row['base64']

    # Add padding to the Base64 string if required
    missing_padding = len(base64_encoded) % 4
    if missing_padding != 0:
        base64_encoded += '=' * (4 - missing_padding)

    try:
        # Decode the Base64 string back to an image
        image_data = base64.b64decode(base64_encoded)

        # Convert the image data to a numpy array and save using OpenCV
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image_array is None:
            print("Failed to decode the image.")
            return

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Write the image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image_array)

        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error decoding the Base64 string: {e}")


        

if __name__ == "__main__":
    # Define input folder, output CSV, and output folder
    input_folder = "train_data/scorecard"
    output_csv = "train_data/images_base64.csv"
    output_folder = "./"


    # Convert images to base64 and save to CSV
    # images_to_base64_csv(input_folder, output_csv)

    # Decode the first image from CSV and save it
    base64_to_image(output_csv, output_folder)
