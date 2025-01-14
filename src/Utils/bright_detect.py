
import sys
sys.path.append("")

import cv2
import os
from typing import Optional

def detect_bright_spots(
    image_path: str,
    output_thresh_path: Optional[str] = None,
    output_result_path: Optional[str] = None,
    contour_area_threshold: int = 300,
    std_multiplier: float = 1.0,
    min_dynamic_threshold: int = 240,
    area_ratio: float = 0.05
) -> bool:
    """
    Detect bright spots in an image and optionally save the thresholded and processed results.

    Parameters:
    - image_path (str): Path to the input image file.
    - output_thresh_path (Optional[str]): Path to save the thresholded image (if provided).
    - output_result_path (Optional[str]): Path to save the final processed image with contours (if provided).
    - contour_area_threshold (int): Minimum contour area to consider for highlighting.
    - std_multiplier (float): Multiplier for the standard deviation to determine dynamic threshold.
    - min_dynamic_threshold (int): Minimum threshold value to apply.
    - area_ratio (float): Ratio of the total image area to consider as a large bright spot.

    Returns:
    - bool: True if a bright spot is detected, False otherwise.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate mean intensity and dynamic threshold
    mean_intensity = gray.mean()
    dynamic_threshold = mean_intensity + std_multiplier * gray.std()
    dynamic_threshold = max(dynamic_threshold, min_dynamic_threshold)

    print("Dynamic threshold:", dynamic_threshold)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, dynamic_threshold, 255, cv2.THRESH_BINARY)

    # Save thresholded image if path is provided
    if output_thresh_path:
        os.makedirs(os.path.dirname(output_thresh_path), exist_ok=True)
        cv2.imwrite(output_thresh_path, thresh)

    # Noise reduction with morphological operations
    thresh = cv2.erode(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]

    bright_spot_detected = False

    # Process each contour
    for i, c in enumerate(contours):
        if cv2.contourArea(c) > contour_area_threshold:
            bright_spot_detected = True
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
            cv2.putText(image, f"#{i + 1}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Check for large bright spots
    for c in contours:
        if cv2.contourArea(c) >= area_ratio * image_area:
            bright_spot_detected = True
            print("Large bright spot detected. Skipping further processing.")

    # Save the result image if path is provided
    if output_result_path:
        os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
        cv2.imwrite(output_result_path, image)

    return bright_spot_detected

if __name__ == "__main__":
    test_image_path = 'images/loa_1.png'
    output_thresh_path = 'docs/thresh_test.jpg'
    output_result_path = 'docs/bright_spots_test.jpg'
    result = detect_bright_spots(test_image_path, output_thresh_path, output_result_path)
    print(f"Bright spot detected: {result}")

