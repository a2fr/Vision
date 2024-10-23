import cv2
import numpy as np
import imutils
import config
import os

def detect_floor(image):
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create a mask that passes the bottom 4/5 of the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Let the bottom 4/5 of the image pass by filling that region with white
    bottom_start = height // 5  # Starting point for the bottom 4/5
    mask[bottom_start:, :] = 255

    return mask

def detect_changes_absdiff(image_reference_path, images_changed_path):
    # Load the reference image
    image_reference_raw = cv2.imread(image_reference_path)
    image_reference = imutils.resize(image_reference_raw, width=800)

    # Load the changed image
    image_changed_raw = cv2.imread(images_changed_path)
    image_changed = imutils.resize(image_changed_raw, width=800)

    # Detect the floor in both images
    floor_mask_ref = detect_floor(image_reference)
    floor_mask_changed = detect_floor(image_changed)

    # Apply the masks to isolate the floor
    floor_ref = cv2.bitwise_and(image_reference, image_reference, mask=floor_mask_ref)
    floor_changed = cv2.bitwise_and(image_changed, image_changed, mask=floor_mask_changed)

    # Convert the images to grayscale
    gray_ref = cv2.cvtColor(floor_ref, cv2.COLOR_BGR2GRAY)
    gray_changed = cv2.cvtColor(floor_changed, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for better contrast
    gray_ref = cv2.equalizeHist(gray_ref)
    gray_changed = cv2.equalizeHist(gray_changed)

    # Calculate the absolute difference between the two images
    diff = cv2.absdiff(gray_ref, gray_changed)
    cv2.imshow("jgl diff", diff)

    # Apply Gaussian blur to reduce noise
    blurred_diff = cv2.GaussianBlur(diff, (9, 9), 0)
    cv2.imshow("blurred diff", blurred_diff)

    # Threshold to isolate changes
    _, thresh = cv2.threshold(blurred_diff, 95, 255, cv2.THRESH_BINARY)
    #cv2.imshow("thresh", thresh)

    # Clean up small elements with morphological operations
    kernel = np.ones((201, 201), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect contours around changes
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around changes
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_changed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the image with detected changes
    cv2.imshow("Changes Detected", image_changed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
ref_name = "/Reference.JPG" if os.name != 'nt' else "\\Reference.JPG"
img_name = "/IMG_6560.JPG" if os.name != 'nt' else "\\IMG_6568.JPG"
detect_changes_absdiff(config.chambre_path + ref_name, config.chambre_path + img_name)