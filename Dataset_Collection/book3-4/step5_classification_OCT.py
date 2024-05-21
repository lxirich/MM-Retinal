import cv2
import numpy as np
import os
from scipy.spatial import distance
import shutil
from PIL import Image
import imagehash

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def is_grey_scale(img_path, tolerance=50):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if abs(r - g) > tolerance or abs(r - b) > tolerance or abs(g - b) > tolerance:
                return False
    return True

# Define a function to calculate the dhash value of an image
def get_dhash(img_path):
    img = Image.open(img_path)
    dhash = imagehash.dhash(img)
    return dhash

# Read reference images
path = './book3/'
reference_image_CFP = cv2.imread(path + "images/1/figure5-3-1A.jpg") # Replace after initial screening with Kmeans
reference_image_FFA = cv2.imread(path + "images/2/figure5-3-1B.jpg") # Replace after initial screening with Kmeans
reference_image_OCT = cv2.imread(path + "images/0/figure5-3-44.jpg") # Replace after initial screening with Kmeans

# Extract color histograms of the reference images
reference_hist_red = extract_color_histogram(reference_image_CFP)
reference_hist_grey = extract_color_histogram(reference_image_FFA)
reference_hist_OCT = extract_color_histogram(reference_image_OCT)

# Read the images to be classified
image_folder = path + 'images/2/' # Replace after initial screening with Kmeans
image_files = os.listdir(image_folder)

# Folder for classified files
OCT_dir = image_folder + "OCT/"
if not os.path.exists(OCT_dir):
    os.mkdir(OCT_dir)

for image_file in image_files:
    if image_file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        src_file = os.path.join(image_folder, image_file)
        # Extract color histogram of the image to be classified
        image = cv2.imread(os.path.join(image_folder, image_file))
        hist = extract_color_histogram(image)

        # Calculate the distance between the image to be classified and the reference images' color histograms
        dist1 = distance.euclidean(hist, reference_hist_red)
        dist2 = distance.euclidean(hist, reference_hist_grey)
        dist3 = distance.euclidean(hist, reference_hist_OCT)

        # Classify based on the distance of color histograms
        if dist3 < dist2 and dist3 < dist1:
            print(f'{image_file} is more similar to image 1')
            shutil.copy(src_file, OCT_dir + image_file)
        else:
            pass
