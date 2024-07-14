import os
import shutil
import tqdm

source_folder = r"images"
labels_folder = r"labels"
output_folder = 'output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of files in the source folder
files = os.listdir(source_folder)

for file in tqdm.tqdm(files):
    # Get the file name without extension
    file_name = os.path.splitext(file)[0]

    # Construct the paths for the source image and label file
    image_path = os.path.join(source_folder, file)
    label_path = os.path.join(labels_folder, file_name + '.txt')

    # Check if the label file exists
    if os.path.exists(label_path):
        # Read the contents of the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Modify the class label to 0
        ans = []
        for line in lines:
            new_line = line.split()
            if new_line[0]!="0":
                print('Found a label other than 0, changing to 0', new_line)
            new_line[0] = '0'
            ans.append(' '.join(new_line))
        modified_lines =ans

        # Write the modified label file to the output folder
        # output_label_path = os.path.join(output_folder, file_name + '.txt')
        with open(label_path, 'w') as f:
            f.writelines(modified_lines)

        # Copy the source image to the output folder
        # output_image_path = os.path.join(output_folder, file)
        # shutil.copy(image_path, output_image_path)
    else:
        print(f"Label file not found for {file_name}")

print("Labels changed successfully!")