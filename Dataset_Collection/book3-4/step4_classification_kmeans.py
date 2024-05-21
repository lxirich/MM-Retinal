import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def classify_by_Kmeans(image_folder):
    # Read all .png and .jpg images in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png','.jpg'))]

    # Extract color histograms of all images
    histograms = []
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_folder, image_file))
        hist = extract_color_histogram(image)
        histograms.append(hist)

    # Perform image classification using KMeans
    kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    kmeans.fit(histograms)

    # Move images to corresponding subfolders based on classification results
    for i, label in enumerate(kmeans.labels_):
        # Create subfolder (if it does not exist)
        sub_folder = os.path.join(image_folder, str(label))
        os.makedirs(sub_folder, exist_ok=True)

        # Move image to subfolder
        src_file = os.path.join(image_folder, image_files[i])
        dst_file = os.path.join(sub_folder, image_files[i])
        shutil.copy(src_file, dst_file)


if __name__=="__main__":
    # Path to the folder containing images to be classified
    original_image_folder = './book3/images'  # Replace

    # Use KMeans to perform initial image screening
    classify_by_Kmeans(original_image_folder)
