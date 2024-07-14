import os
import random
import shutil

def random_select_images(input_folder, output_folder, n):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # Randomly select n images
    selected_images = random.sample(image_files, n)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)

    # Copy the selected images and labels to the output folder
    for image_file in selected_images:
        shutil.copy(os.path.join(input_folder, image_file), os.path.join(output_folder, 'images'))
        label_file = os.path.splitext(image_file)[0] + '.txt'
        shutil.copy(os.path.join(input_folder_labels, label_file), os.path.join(output_folder, 'labels'))

    print(f"Successfully selected and saved {n} images in the YOLO format.")

# Usage example
input_folder = r"images" #path to image folder. Change accordingly
input_folder_labels = r"labels" #path to labels folder. Change accordingly
output_folder = r"sample"
n = 300  # Number of images to randomly select

random_select_images(input_folder, output_folder, n)