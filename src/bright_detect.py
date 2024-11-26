import sys
sys.path.append("")

import cv2


image = cv2.imread('images/loa_1.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to detect bright spots
# Calculate the mean intensity of the image
mean_intensity = gray.mean()

# Define the dynamic threshold (80% above the mean)
dynamic_threshold = mean_intensity + 1 * gray.std()
if dynamic_threshold < 240:
    dynamic_threshold = 240


print("dynamic_threshold", dynamic_threshold)
_, thresh = cv2.threshold(gray, dynamic_threshold, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh 1", thresh)

# Perform a series of erosions and dilations to clean up small noise blobs
thresh = cv2.erode(thresh, None, iterations=2)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the total image area
image_area = gray.shape[0] * gray.shape[1]


# Loop over the contours to draw them on the original image
for i, c in enumerate(contours):
    # Only consider large enough contours (bright spots)
    if cv2.contourArea(c) > 300:  # Adjust the threshold as needed
        # Get the bounding box and center of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        
        # Draw a circle around the bright spot
        cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        
        # Annotate with the contour number
        cv2.putText(image, "#{}".format(i + 1), (x, y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Check if any contour area is >= 10% of the image area
for c in contours:
    if cv2.contourArea(c) >= 0.05 * image_area:
        print("Large bright spot detected. Skipping further processing.")
        
# Show the output image with drawn contours and bright spots
cv2.imshow("Bright Spots", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

