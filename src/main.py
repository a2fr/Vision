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

# Function to preprocess the image: resize, apply masks, convert to grayscale, and equalize histograms
def preprocess_image(image_path):
    image_raw = cv2.imread(image_path)
    image_resized = imutils.resize(image_raw, width=800)

    # Detect the floor and apply the floor mask
    floor_mask = detect_floor(image_resized)
    floor_image = cv2.bitwise_and(image_resized, image_resized, mask=floor_mask)

    # Convert the floor area to grayscale and equalize the histogram
    gray_image = cv2.cvtColor(floor_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)

    return equalized_image

# Function to calculate absolute difference and apply Gaussian blur
def calculate_absdiff(image_ref, image_changed):
    diff = cv2.absdiff(image_ref, image_changed)
    blurred_diff = cv2.GaussianBlur(diff, (3, 3), 0)
    return blurred_diff

# Function to apply thresholding to isolate significant differences
def apply_threshold(blurred_diff, threshold_value):
    _, thresholded_diff = cv2.threshold(blurred_diff, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_diff

# Function to apply Canny edge detection and clean up the edges with morphological operations
def detect_edges(thresholded_diff):
    edges = cv2.Canny(thresholded_diff, 150, 55)
    kernel = np.ones((9, 9), np.uint8)  # Smaller kernel to preserve details
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return cleaned_edges

# Function to find and draw bounding boxes around detected contours
def draw_contours(image_changed, contours):
    for contour in contours:
        if cv2.contourArea(contour) > 1200:  # Adjust threshold for object size
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_changed, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_changed

# Combined function that calls all helper functions and displays the detected changes
def detect_changes(image_reference_path, image_changed_path, threshold_value):
    # Preprocess both reference and changed images
    gray_ref = preprocess_image(image_reference_path)
    gray_changed = preprocess_image(image_changed_path)

    # Calculate absolute difference and blur
    blurred_diff = calculate_absdiff(gray_ref, gray_changed)

    # Apply thresholding to isolate significant differences
    thresholded_diff = apply_threshold(blurred_diff, threshold_value)

    # Perform edge detection and clean edges
    cleaned_edges = detect_edges(thresholded_diff)

    # Detect contours in the cleaned edge image
    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load and resize the changed image to draw bounding boxes
    image_changed_raw = cv2.imread(image_changed_path)
    image_changed = imutils.resize(image_changed_raw, width=800)

    # Draw bounding boxes around detected contours
    image_with_contours = draw_contours(image_changed, contours)

    # Display the result
    cv2.imshow("Detected Changes", image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
ref_name = "/Reference.JPG" if os.name != 'nt' else "\\Reference.JPG"
img_name = "/IMG_6560.JPG" if os.name != 'nt' else "\\IMG_6557.JPG"
threshold_value = 95  # Adjust this value based on your needs
detect_changes(config.salon_path + ref_name, config.salon_path + img_name, threshold_value)
