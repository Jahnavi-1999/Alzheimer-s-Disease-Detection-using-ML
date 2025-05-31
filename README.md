# MindSight Diagnosis - Backend (Alzheimer Detection)

This is the Flask backend for Alzheimer's disease stage detection using MRI or PET scan images.
It uses a trained CNN model and Grad-CAM visualization.

## Folder Structure

- model/ : Contains the trained brain_cnn_model.h5 file.
- plots/ : Contains training plots (accuracy and loss graphs).
- app.py : Flask server to handle predictions and Grad-CAM requests.
- gradcam.py : Helper script to generate Grad-CAM heatmaps.
- utils.py : Image preprocessing helpers.
- test_client_predict.py : Test script to predict disease stage.
- test_client_gradcam.py : Test script to generate Grad-CAM heatmap.
- requirements.txt : List of required Python libraries.

## Setup Instructions

1. Navigate to backend folder:
```bash
cd backend
