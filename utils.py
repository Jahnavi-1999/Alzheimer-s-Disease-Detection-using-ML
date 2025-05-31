import numpy as np
from PIL import Image

def preprocess_image(file):
    # Read the uploaded file into an image
    image = Image.open(file)

    # Convert image to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 128x128 (the size expected by the CNN)
    image = image.resize((128, 128))

    # Convert image to numpy array
    image = np.array(image)

    # Normalize pixel values to [0,1]
    image = image / 255.0

    # Expand dims to add channel axis (128,128) -> (128,128,1)
    image = np.expand_dims(image, axis=-1)

    return image
