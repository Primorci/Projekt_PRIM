import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np

# Load the models
modelDanger = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/mihap/yolov5/runs/train/exp9/weights/best.pt')
modelRoad = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo\RoadBest.pt')

def resize_image(image, max_size=(800, 600)):
    """
    Resize the image to fit within a specific size while maintaining the aspect ratio.
    """
    h, w = image.shape[:2]
    ratio = min(max_size[0] / w, max_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image

def open_video():
    """
    Open a video file using a file dialog and start object detection.
    """
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        window.title(f"Danger on the Road Detection - {file_path}")
        detect_objects()
    else:
        messagebox.showinfo("Information", "No video file selected.")

def detect_objects():
    """
    Detect objects in the video frame by frame and display the results on the canvas.
    """
    global cap, canvas, window, photo
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the color from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection using both models
            results_danger = modelDanger(frame)
            results_road = modelRoad(frame)

            # Render detections
            results_danger.render()
            results_road.render()

            # Update the detection results label
            detected_classes = []
            if results_danger.pred[0] is not None:
                detected_classes.extend(results_danger.names[int(cls)] for cls in results_danger.pred[0][:, -1])
            if results_road.pred[0] is not None:
                detected_classes.extend(results_road.names[int(cls)] for cls in results_road.pred[0][:, -1])

            # Convert array to Image
            frame_image = Image.fromarray(frame)
            frame_resized = resize_image(np.array(frame_image))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            canvas.create_image(20, 20, anchor='nw', image=photo)
            window.after(64, detect_objects)
        else:
            cap.release()

# Create the main window
window = tk.Tk()
window.title("Detection App")
window.geometry("820x680")  # Adjust window size to include status bar

# Create a canvas to show the video frames
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Buttons for loading video and running detection
btn_load = tk.Button(window, text="Open Video", command=open_video)
btn_load.pack(side='left')

window.mainloop()