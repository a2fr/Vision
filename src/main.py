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

def detect_changes_absdiff(image_reference_path, images_changed_path, image_name):
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
    blurred_diff = cv2.GaussianBlur(diff, (9, 9), 0)

    # Threshold to isolate changes
    _, thresh = cv2.threshold(blurred_diff, 95, 255, cv2.THRESH_BINARY)

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
    cv2.imshow(image_name, image_changed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_folder(folder_path):
    # Get the reference image path
    ref_name = "/Reference.JPG" if os.name != 'nt' else "\\Reference.JPG"
    image_reference_path = folder_path + ref_name

    # Get all changed images in the folder
    changed_images = config.get_changed_images(folder_path)

    # Process each changed image
    for img_name in changed_images:
        images_changed_path = os.path.join(folder_path, img_name)
        detect_changes_absdiff(image_reference_path, images_changed_path, img_name)

if __name__ == "__main__":
    # Select the folder to process
    folder_selection = input("Select the folder to process (chambre, salon, cuisine): ").strip().lower()

    if folder_selection == "chambre":
        folder_path = config.chambre_path
    elif folder_selection == "salon":
        folder_path = config.salon_path
    elif folder_selection == "cuisine":
        folder_path = config.cuisine_path
    else:
        print("Invalid selection. Please choose from chambre, salon, or cuisine.")
        exit(1)

    # Process the selected folder
    process_folder(folder_path)
