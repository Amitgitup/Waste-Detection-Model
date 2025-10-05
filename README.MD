# Waste Detection Model with YOLOv8

A Flask-based web application that uses YOLOv8 to detect and classify different types of waste items in images. The application supports both file upload and real-time webcam detection.

## 🎯 Project Overview

This project implements a waste detection system that can identify 13 different types of waste items:
- banana, chilli, drinkcan, drinkpack, foodcan, lettuce, paperbag, plasticbag, plasticbottle, plasticcontainer, sweetpotato, teabag, tissueroll

The system uses YOLOv8 (You Only Look Once) for real-time object detection and provides a user-friendly web interface for both image upload and live webcam detection.

## 🚀 Features

- **Image Upload Detection**: Upload images and get instant waste detection results
- **Real-time Webcam Detection**: Use your webcam for live waste detection
- **13 Waste Categories**: Detects various types of waste items
- **Modern Web Interface**: Clean, responsive UI built with Tailwind CSS
- **Flask Backend**: Robust Python backend with error handling
- **YOLOv8 Integration**: State-of-the-art object detection model

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package installer)
- A webcam (for real-time detection feature)

## 🛠️ Installation & Setup

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd "Waste Detection Model"

# Or download and extract the project files to your desired directory
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate

# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install flask ultralytics torch numpy opencv-python
```

### Step 4: Download Pre-trained Model

The application requires a trained YOLOv8 model. You have two options:

#### Option A: Use the provided model (if available)
Ensure the model file exists at: `runs/detect/yolov8n_waste_detection/weights/model.pt`

#### Option B: Train your own model
Follow the training steps in the `Waste_Detection_Training.ipynb` notebook.

### Step 5: Verify Dataset Structure

Ensure your dataset follows this structure:
```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

## 🎓 Step-by-Step Solution Approach

### Phase 1: Data Preparation
1. **Collect Dataset**: Gather images of waste items across all 13 categories 
2. **Label Images**: Use tools like LabelImg or Roboflow to create YOLO format annotations
3. **Split Dataset**: Divide into train/validation sets (typically 80/20 split)
4. **Create data.yaml**: Configure dataset paths and class names

### Phase 2: Model Training
1. **Choose Base Model**: Start with YOLOv8n.pt for faster training
2. **Configure Training Parameters**:
   - Epochs: 20-100 (start with 20)
   - Batch size: 16 (adjust based on GPU memory)
   - Image size: 640x640
3. **Train Model**: Use the provided Jupyter notebook
4. **Validate Results**: Check mAP scores and detection accuracy

### Phase 3: Web Application Development
1. **Flask Backend**: Create API endpoints for image processing
2. **Frontend Interface**: Build responsive UI with upload and webcam features
3. **Model Integration**: Load trained YOLOv8 model for inference
4. **Error Handling**: Implement robust error handling and user feedback

### Phase 4: Testing & Optimization
1. **Test with Various Images**: Validate detection accuracy
2. **Performance Optimization**: Optimize for speed and accuracy
3. **User Experience**: Refine UI/UX based on testing feedback

## 🚀 Running the Application

### Start the Flask Server

```bash
# Make sure you're in the project directory and virtual environment is activated
python app.py
```

The application will start on `http://localhost:5000` or `http://0.0.0.0:5000`

### Access the Web Interface

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. You'll see the waste detection interface with two options:
   - **Upload Image**: Click to upload and detect waste in images
   - **Webcam Detection**: Click "Start Webcam" for real-time detection

## 📁 Project Structure

```
Waste Detection Model/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── yolov8n.pt                     # Pre-trained YOLOv8 model
├── Waste_Detection_Training.ipynb # Training notebook
├── dataset/                       # Training dataset
│   ├── data.yaml                  # Dataset configuration
│   ├── train/                     # Training images and labels
│   └── valid/                     # Validation images and labels
├── templates/
│   └── index.html                 # Web interface template
├── uploads/                       # Uploaded images storage
├── processed/                     # Processed images with detections
├── runs/                          # Training outputs and model weights
└── myenv/                         # Virtual environment
```

## 🔧 Configuration

### Model Path Configuration

Update the model path in `app.py` if your trained model is located elsewhere:

```python
MODEL_PATH = 'path/to/your/trained/model.pt'
```

### Dataset Configuration

Modify `dataset/data.yaml` to match your dataset structure:

```yaml
train: ../train/images
val: ../valid/images

nc: 13  # Number of classes
names: ['banana', 'chilli', 'drinkcan', 'drinkpack', 'foodcan', 'lettuce', 'paperbag', 'plasticbag', 'plasticbottle', 'plasticcontainer', 'sweetpotato', 'teabag', 'tissueroll']
```

## 🎯 Usage Examples

### File Upload Detection
1. Click on the upload area
2. Select an image file (PNG, JPG, JPEG)
3. Wait for processing
4. View original and detected images side by side

### Webcam Detection
1. Click "Start Webcam" (grant camera permission)
2. Position waste items in front of the camera
3. Click "Capture & Predict"
4. View detection results

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure the model file exists at the specified path
   - Check file permissions
   - Verify the model file is not corrupted

2. **Webcam Not Working**:
   - Grant camera permissions in your browser
   - Ensure no other applications are using the camera
   - Try refreshing the page

3. **Poor Detection Accuracy**:
   - Train the model with more epochs
   - Use a larger dataset
   - Try different YOLOv8 model sizes (s, m, l, x)

4. **Slow Performance**:
   - Reduce image size in training
   - Use GPU acceleration if available
   - Optimize batch size

### Performance Optimization

1. **GPU Acceleration**: Install CUDA-compatible PyTorch for faster inference
2. **Model Size**: Use YOLOv8s/m/l/x for better accuracy (slower)
3. **Image Resolution**: Reduce input image size for faster processing

## 📈 Future Improvements

Based on the training notebook, here are potential improvements:

1. **Better Dataset**: Collect more diverse and high-quality images
2. **Advanced Model**: Use YOLOv8x.pt for better accuracy
3. **GPU Training**: Train on GPU (Kaggle/Google Colab) for faster training
4. **More Epochs**: Increase training epochs (50-100+)
5. **Data Augmentation**: Apply advanced augmentation techniques
6. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
7. **Modularised Coding**: Coulda Modularize everything properly...()

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for styling

---
**Happy Waste Detection! 🌱♻️**


**There's so much SCOPE of improvement... will try to work on it...I am up for the discussions..**

**As of now, I am working on how to integrate 'MAPMYINDIA' in the project...**

