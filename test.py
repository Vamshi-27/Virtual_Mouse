import os
import shutil

# Create the folder if it doesn't exist
output_folder = "data/thumbs_up"
os.makedirs(output_folder, exist_ok=True)

# Copy with continuous numbering
for i in range(2, 501):  # From 2 to 500
    destination = os.path.join(output_folder, f"{i}.png")
    shutil.copy("data/thumbs_up/1.png", destination)
