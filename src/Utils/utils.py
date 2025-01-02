import sys
sys.path.append("")

from functools import wraps
import time
import os
import yaml
from PIL import Image
from pytesseract import Output
import pytesseract
from collections import Counter
import cv2
import numpy as np
import requests
import warnings
import contextlib
from urllib3.exceptions import InsecureRequestWarning


from dotenv import load_dotenv
load_dotenv()


# SSL Verification Context Manager
@contextlib.contextmanager
def no_ssl_verification():
    old_merge_environment_settings = requests.Session.merge_environment_settings
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass



def convert_ms_to_hms(ms):
    seconds = ms / 1000
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    seconds = round(seconds, 2)
    
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        time_delta = convert_ms_to_hms(total_time*1000)

        print(f'{func.__name__.title()} Took {time_delta}')
        return result
    return timeit_wrapper

def is_file(path:str):
    return '.' in path

def check_path(path):
    # Extract the last element from the path
    last_element = os.path.basename(path)
    if is_file(last_element):
        # If it's a file, get the directory part of the path
        folder_path = os.path.dirname(path)

        # Check if the directory exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create new folder path: {folder_path}")
        return path
    else:
        # If it's not a file, it's a directory path
        # Check if the directory exists, create it if not
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create new path: {path}")
        return path

def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def resize_same_ratio(img: Image.Image, target_size: int = 640) -> Image.Image:
    """
    Resizes an image to maintain aspect ratio either by height or width.
    
    If the image is vertical (height > width), it resizes the image to make its height 640 pixels,
    maintaining the aspect ratio. If the image is horizontal (width > height), it resizes the image 
    to make its width 640 pixels, maintaining the aspect ratio.
    
    Args:
        img (PIL.Image.Image): The image to be resized.
        target_size (int, optional): The target size for the longest dimension of the image. 
                                     Defaults to 640.

    Returns:
        PIL.Image.Image: The resized image.
    """
    
    # Get the current dimensions of the image
    width, height = img.size
    
    # Determine whether the image is vertical or horizontal
    if height > width:
        # Calculate the new width to maintain aspect ratio
        new_width = int((width / height) * target_size)
        new_height = target_size
    else:
        # Calculate the new height to maintain aspect ratio
        new_height = int((height / width) * target_size)
        new_width = target_size

    # Resize the image
    resized_img = img.resize((new_width, new_height))
    
    return resized_img


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def get_rotation_angle(img: Image.Image) -> int:
    try:
        """
        Gets the rotation angle of the image using Tesseract's OSD.

        Args:
            img (PIL.Image.Image): The image to analyze.

        Returns:
            int: The rotation angle.
        """
        # Convert PIL image to OpenCV format
        image_cv = np.array(img)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Use pytesseract to get orientation information
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT, config='--psm 0 -c min_characters_to_try=5')
        
        return results["rotate"]
    except Exception as e:
        return 0
    
def rotate_image(img: Image.Image) -> Image.Image:
    """
    Rotates an image to correct its orientation based on the detected rotation angle
    by analyzing the image at different sizes and choosing the most frequent angle.
    
    Args:
        img (PIL.Image.Image): The image to be rotated.

    Returns:
        PIL.Image.Image: The rotated image.
    """
    
    # Resize the image to different target sizes
    target_sizes = [640, 1080, 2000]
    rotation_angles = []

    for size in target_sizes:
        resized_img = resize_same_ratio(img, target_size=size)
        rotation_angle = get_rotation_angle(resized_img)
        rotation_angles.append(rotation_angle)

    # Find the most common rotation angle
    most_common_angle = Counter(rotation_angles).most_common(1)[0][0]

    if abs(most_common_angle) in [0, 180]:
        return img, 0

    # Rotate the original image using the most common angle
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    rotated = rotate_bound(image_cv, angle=most_common_angle)
    
    # Convert the rotated image back to PIL format
    rotated_pil = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    return rotated_pil, most_common_angle