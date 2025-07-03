from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Path to save uploaded images and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = '/home/Desktop/fracture_detection_project/model/best.pt' #edit here

# Create directories if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')  # Your main page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    img2 = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Read the image
    img = cv2.imread(filepath)
    
    if img is None:
        return "Error loading image", 400
    
    # Preprocess image (optional)
    img = cv2.resize(img, (640, 640))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(gray)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Run YOLO model
    results = model(img)[0]
    detected_boxes = results.boxes.data.tolist()

    # Initialize result variables
    fracture_detected = False

    # Draw bounding boxes if fractures are detected
    for result in detected_boxes:
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if score > 0.3:  # Confidence threshold
            fracture_detected = True
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = results.names[int(class_id)].upper()
            cv2.putText(img, f"{class_name} ({score:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Save the result image
    result_path = os.path.join(RESULT_FOLDER, file.filename)
    cv2.imwrite(result_path, img)

    # Display diagnosis
    diagnosis = "Fracture Detected ❌" if fracture_detected else "No Fracture Detected ✅"

    # Return the result page
    return render_template('result.html', 
                           result_image=result_path,
                           diagnosis=diagnosis,input_image=img2)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)