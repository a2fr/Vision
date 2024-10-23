import cv2
import numpy as np
import imutils
import config


def segment_floor(image, k_clusters=2):
    # Resize image for faster computation if needed
    image_resized = imutils.resize(image, width=800)

    # Convert image to LAB color space for better lighting handling
    lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = cv2.KMeans(n_clusters=k_clusters, random_state=0)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    # Reshape labels back to the original image shape
    labels = labels.reshape(image_resized.shape[:2])

    # Create masks for each cluster
    unique_labels = np.unique(labels)
    masks = []
    for label in unique_labels:
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        masks.append(mask)

    # Optionally, show the segmentation result for manual inspection
    for i, mask in enumerate(masks):
        cv2.imshow(f"Cluster {i}", mask)

    # Return all cluster masks for further processing
    return masks


def detect_changes_absdiff(image_reference_path, images_changed_path):
    # Load the reference image
    image_reference_raw = cv2.imread(image_reference_path)
    image_reference = imutils.resize(image_reference_raw, width=800)

    # Load the changed image
    image_changed_raw = cv2.imread(images_changed_path)
    image_changed = imutils.resize(image_changed_raw, width=800)

    # Segment the floor using K-means clustering for both images
    floor_masks_ref = segment_floor(image_reference)
    floor_masks_changed = segment_floor(image_changed)

    # You can manually select which cluster represents the floor, or apply logic to choose
    # Here, we are assuming that the floor is represented by the first cluster (you can refine this logic)
    floor_mask_ref = floor_masks_ref[0]
    floor_mask_changed = floor_masks_changed[0]

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
detect_changes_absdiff(config.salon_path + "\\Reference.JPG", config.salon_path + "\\IMG_6552.JPG")
