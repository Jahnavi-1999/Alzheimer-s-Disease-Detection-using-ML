import tensorflow as tf
try:
    model = tf.keras.models.load_model('model/brain_cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
