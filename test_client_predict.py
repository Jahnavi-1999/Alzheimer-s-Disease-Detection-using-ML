import requests

# File path to your test MRI image
file_path = "test_mri.jpg"  # <-- Replace with actual image path

url = "http://localhost:5000/predict"

files = {"file": open(file_path, "rb")}

response = requests.post(url, files=files)

print("Response from server:", response.json())
