import torch
from PIL import Image
import os
#Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#Directory containing images
directory = 'TestFrames'
#Loop over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):  # assuming you're interested in .jpg files
        # Load an image
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path)

        # Inference
        results = model(img)

        # Results
        results.print()  # Print results to console
        results.show()   # Display image with bounding boxes
        results.save()   # Save image with bounding boxes to 'runs/detect/exp'
        print(f"Processed {filename}")
    else:
        continue