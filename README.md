
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
![Screenshot 2025-05-20 233754](https://github.com/user-attachments/assets/f4d6125c-4426-4e8c-809f-ea8ac6625ec0)
![Screenshot 2025-05-20 233846](https://github.com/user-attachments/assets/b9447659-5f6a-4128-b9f3-a712cca5ed45)
![Screenshot 2025-05-20 233924](https://github.com/user-attachments/assets/a999748c-6a8e-4fcc-be9f-03b16406e418)
![Screenshot 2025-05-20 234008](https://github.com/user-attachments/assets/f5c63a85-f292-40f9-b093-7d81a415042d)









