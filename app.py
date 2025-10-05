import os
import uuid
import base64
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from ultralytics import YOLO
import cv2

# --- Configuration ---
# IMPORTANT: Update this path to your trained model weights file.
# This file is typically found in 'runs/detect/train/weights/best.pt' after training.
MODEL_PATH = 'runs\detect\yolov8n_waste_detection2\weights\model.pt'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed' # Changed from 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Create Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# --- Create necessary directories ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- Load YOLOv8 Model ---
try:
    model = YOLO(MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # This allows the app to run so the user can see the UI and the error message.
    model = None

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_and_save_image(image_data, original_filename):
    """Runs YOLO inference on an image, saves the original and processed images."""
    # Save the uploaded file with a secure name
    filename = str(uuid.uuid4()) + os.path.splitext(original_filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_data.save(filepath)

    # Perform inference
    results = model(filepath)
    
    # Plot results on the image
    result_image_array = results[0].plot() # This returns a BGR numpy array

    # Save the processed image to the processed folder
    processed_filename = 'processed_' + filename
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    # Use cv2.imwrite for saving numpy arrays
    cv2.imwrite(processed_filepath, result_image_array)

    return filename, processed_filename

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    """Handle file uploads and run YOLOv8 prediction."""
    model_error = None
    if model is None:
        model_error = f"The model file could not be loaded from '{MODEL_PATH}'. Please ensure the path is correct and the file is not corrupted."

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', model_error="No file part in the request.")
        
        file = request.files['file']
        
        if file.filename == '':
             return render_template('index.html', model_error="No file selected.")

        if file and allowed_file(file.filename) and model:
            filename, processed_filename = process_and_save_image(file, file.filename)
            return render_template(
                'index.html',
                filename=filename,
                processed_filename=processed_filename,
                model_error=model_error
            )
        
        return render_template('index.html', model_error=model_error or "Invalid file type.")

    # For GET request, just display the upload page
    return render_template('index.html', model_error=model_error)

@app.route('/webcam', methods=['POST'])
def webcam_predict():
    """Handle webcam image data for prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    image_data = data['image']
    # Decode the base64 string
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    
    # Convert binary data to a numpy array
    nparr = np.frombuffer(binary_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Create a unique filename for the webcam capture
    filename = f"webcam_{uuid.uuid4()}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img_np)

    # Perform inference
    results = model(filepath)
    result_image_array = results[0].plot()

    # Save the processed image
    processed_filename = 'processed_' + filename
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_filepath, result_image_array)

    # Return the URL to the processed image
    return jsonify({'processed_image_url': f'/processed/{processed_filename}'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    """Serve processed files."""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

