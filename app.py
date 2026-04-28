import os

from flask import Flask, request, render_template, send_file

from ultralytics import YOLO

import cv2

import numpy as np

from PIL import Image
import io


app = Flask(__name__, static_folder='static', template_folder='templates')


# Ensure the 'uploads' and 'results' directories exist

UPLOAD_FOLDER = 'uploads'

RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(RESULT_FOLDER, exist_ok=True)


# Load the best trained model weights

# Make sure 'best.pt' is in the same directory as app.py or provide the full path

model = YOLO("C:/100rabh/naman_ak/best.pt") # Adjust path if running locally


@app.route('/')

def index():

    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():

    if 'file' not in request.files:

        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':

        return 'No selected file', 400

    if file:
        filename = file.filename

        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)


        # Perform inference

        # Ensure the model is loaded with the correct path when running locally

        results = model.predict(source=filepath, conf=0.25, save=False) # conf=0.25 sets a confidence threshold


        # Process results and save the annotated image

        result_img_path = os.path.join(RESULT_FOLDER, 'predicted_' + filename)

        for r in results:

            im_array = r.plot()  # plot a BGR numpy array of predictions

            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

            im.save(result_img_path)  # save BGR image

        return send_file(result_img_path, mimetype='image/jpeg')

if __name__ == '__main__':
    # When running locally, ensure the 'best.pt' is in the current directory or specified correctly
    # For now, it's pointing to the Colab path. You'll need to download it from Colab first.
    print("To run this locally, first download your 'best.pt' file from /content/runs/detect/train/weights/best.pt")
    print("Then, ensure it's in the same directory as app.py, or update the path in app.py.")
    print("Access the application at http://127.0.0.1:5000")
    app.run(debug=True)
