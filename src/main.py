import cv2
import numpy as np
import imutils
import config
import os

def detect_floor(image):
    # Convert image to LAB color space for better lighting robustness
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Perform CLAHE (Contrast Limited Adaptive Histogram Equalization) to reduce lighting variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    hsv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # Define a broad range for floor color detection
    lower_floor = np.array([0, 0, 0])
    upper_floor = np.array([180, 255, 60]) #[180, 255, 60]

    # Create a mask for the floor color
    mask_color = cv2.inRange(hsv, lower_floor, upper_floor)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel)

    # Use adaptive thresholding to segment texture regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_texture = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # Ensure masks are the same size
    mask_texture = cv2.resize(mask_texture, (mask_color.shape[1], mask_color.shape[0]))

    # Combine color and texture masks
    combined_mask = cv2.bitwise_and(mask_color, mask_texture)
    cv2.imshow("Combined Mask", combined_mask)

    return combined_mask

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

    # Apply Gaussian blur to reduce noise
    blurred_diff = cv2.GaussianBlur(diff, (91, 91), 0)

    # Threshold to isolate changes
    _, thresh = cv2.threshold(blurred_diff, 60, 255, cv2.THRESH_BINARY)

    # Clean up small elements with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect contours around changes
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around changes
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_changed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the image with detected changes
    cv2.imshow("Changes Detected", image_changed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
ref_name = "/Reference.JPG" if os.name != 'nt' else "\\Reference.JPG"
img_name = "/IMG_6552.JPG" if os.name != 'nt' else "\\IMG_6552.JPG"
detect_changes_absdiff(config.salon_path + ref_name, config.salon_path + img_name)