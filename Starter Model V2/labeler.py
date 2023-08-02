import os
import re
import json

# Specify the folder path containing the images
folder_path = "C:/Users/Skylar/Desktop/New fake eggs(not overlapping and two diff sizes)"

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter only image files (adjust the extensions as needed)
image_files = [file for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))]

# Create a dictionary to store the image data
image_data = {}

# Define a regular expression pattern to extract the number of dots from the file name
pattern = r"^(\d+)-dots"

# Populate the dictionary with image data
for i, file_name in enumerate(image_files):
    # Extract the number of dots from the file name
    match = re.search(pattern, file_name)
    if match:
        num_dots = int(match.group(1))
    else:
        num_dots = 0
    
    # Assign the image data to the dictionary
    image_data[f"image{i+1}"] = {
        "file_name": file_name,
        "num_dots": num_dots
    }

# Save the dictionary as JSON with indentation
with open("C:/Users/Skylar/Documents/VSCode Projects/Python Projects/eggCounter/labels(no-overlap-2sizes).json", "w") as json_file:
    json.dump(image_data, json_file, indent=2)