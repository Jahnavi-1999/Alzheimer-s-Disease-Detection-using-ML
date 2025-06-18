
#  Alzheimer's Disease Diagnosis System
Alzheimer’s Disease Diagnosis System is an AI-powered web application designed to assist in the early detection and classification of Alzheimer’s Disease using brain MRI scans. Built with a custom Convolutional Neural Network (CNN) and deployed via Streamlit, the tool provides real-time predictions, confidence scores, and doctor-style recommendations — all through a user-friendly interface. This system supports healthcare professionals and caregivers by offering a fast, privacy-compliant, and accessible diagnostic aid for Alzheimer’s care.

## Features
-  Predicts Alzheimer’s stage: `Non-Demented`, `Very Mild Demented`, `Mild Demented`, `Moderate Demented`
-  Confidence scores and prediction probability table
-  Doctor-style recommendation based on results
-  Downloadable diagnosis report (.txt)
-  Real-time predictions with **no data storage** (privacy-compliant)
-  Streamlit-based interactive interface

##  Tech Stack
- Frontend: Streamlit
- Backend: TensorFlow + Keras (CNN model)
- Image Processing: OpenCV, Pillow
- Data Handling: NumPy, Pandas
- Optional Backend API: Flask (preparation for future expansion)

##  Project Structure
 Alzheimer's Diagnosis System
├── app.py # Main Streamlit app
├── model/
│ └── brain_cnn_model.h5 # Trained CNN model
├── utils.py # Image preprocessing functions
├── test_client_predict.py # Test script to validate model output
├── gradcam.py # Optional Grad-CAM script for explainability
├── requirements.txt # Python dependencies
├── README.md # Project documentation

## Model Overview
- Input: MRI brain scan (grayscale, 128x128 px)
- Output: Alzheimer’s stage + confidence score
- Architecture: Custom CNN with Conv, Pool, Dropout layers
- Activation: Softmax for multi-class classification
- Dataset: Publicly available Kaggle MRI scans (sourced from ADNI)
## outputs
![Screenshot 2025-05-20 233704](https://github.com/user-attachments/assets/57658473-74ab-4c71-b4c6-edfcc4ecd378)


## Setup Instructions
# 1. Clone the repository
git clone https://github.com/your-username/AD-detection-using-DL.git
cd AD-detection-using-DL
# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
# 3. Install required dependencies
pip install -r requirements.txt
# 4. Ensure the trained CNN model is in the 'model/' folder
# File required: brain_cnn_model.h5
# 5. Run the Streamlit app
streamlit run app.py


