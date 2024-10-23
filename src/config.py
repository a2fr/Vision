import os

chambre_path = "../data/raw/images/chambre" if os.name != 'nt' else "..\\data\\raw\\images\\chambre"
cuisine_path = "../data/raw/images/cuisine" if os.name != 'nt' else "..\\data\\raw\\images\\cuisine"
salon_path = "../data/raw/images/salon" if os.name != 'nt' else "..\\data\\raw\\images\\salon"

# Function to get all changed images in the chambre folder (excluding the reference)
def get_changed_images(folder):
    all_files = os.listdir(folder)
    changed_images = [f for f in all_files if "Reference" not in f and f.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
    return changed_images
