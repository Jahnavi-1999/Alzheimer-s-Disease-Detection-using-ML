import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from utils import preprocess_image
import os


# --------------------------- Page Setup --------------------------- #
st.set_page_config(page_title="Alzheimer's Disease Diagnosis System", page_icon="🧠", layout="centered")
st.title("🧠 Alzheimer's Disease Diagnosis System")
st.metric(label=" Model Validation Accuracy", value="88.5%")
st.markdown("<h4 style='text-align: center; color: grey;'>AI-Powered Early Detection of Alzheimer's Disease</h4>", unsafe_allow_html=True)
st.markdown("---")


# --------------------------- Load Model --------------------------- #
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "brain_cnn_model.h5")
    model = tf.keras.models.load_model(model_path)

    inputs = tf.keras.Input(shape=(128, 128, 1), name="new_input_layer")
    x = model(inputs)
    functional_model = tf.keras.Model(inputs, x)
    dummy_input = tf.zeros((1, 128, 128, 1), dtype=tf.float32)
    _ = functional_model(dummy_input)
    return functional_model

model = load_model()
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# --------------------------- Upload MRI --------------------------- #
st.subheader("📤 Upload MRI Scan")
uploaded_file = st.file_uploader("Upload a brain MRI image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='🖼️ Uploaded MRI Image', use_container_width=True)


    with st.spinner('🔎 Processing image... Please wait'):
        processed_image = preprocess_image(uploaded_file)
        image_batch = np.expand_dims(processed_image, axis=0)
        image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    st.subheader("🧠 Predict Alzheimer's Condition")

    if st.button('🔮 Predict'):
        with st.spinner('🧠 Predicting condition...'):
            try:
                prediction = model.predict(image_batch)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence Level: **{confidence:.2f}%**")

                # 📊 Prediction Probabilities
                st.subheader("📊 Prediction Probabilities")
                df = pd.DataFrame({
                    'Condition': class_names,
                    'Confidence (%)': [f"{p*100:.2f}" for p in prediction[0]]
                })
                st.table(df)

                # 🩺 Doctor's Suggestion
                st.subheader("🩺 Doctor's Suggestion")
                if predicted_class == 'MildDemented':
                    advice = "Early signs of dementia detected. Recommend full neurological evaluation and lifestyle adjustments."
                    st.warning(advice)
                elif predicted_class == 'ModerateDemented':
                    advice = "Moderate dementia detected. Immediate clinical intervention and medication support are recommended."
                    st.error(advice)
                elif predicted_class == 'NonDemented':
                    advice = "No signs of dementia detected. Maintain a healthy brain lifestyle with exercise, diet, and cognitive activities."
                    st.success(advice)
                elif predicted_class == 'VeryMildDemented':
                    advice = "Very mild symptoms detected. Recommend periodic monitoring and brain health exercises."
                    st.info(advice)
                else:
                    advice = "Further evaluation is recommended to confirm diagnosis."
                    st.info(advice)

                # 📄 Download Report
                st.subheader("📄 Download Doctor Report")
                report = f"""
Mind-Sight Diagnosis System - Doctor's Report
----------------------------------------------

Prediction: {predicted_class}
Confidence: {confidence:.2f}%

Doctor's Advice:
{advice}

Disclaimer: This AI system is for assistive diagnosis only. Please consult a qualified medical professional for a final clinical diagnosis.
"""
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name="Doctor_Report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --------------------------- Disclaimer --------------------------- #
st.markdown("---")
st.markdown("<small><i>Disclaimer: This AI diagnosis system is for educational and assistive purposes only. Final diagnosis must be confirmed by a licensed medical professional.</i></small>", unsafe_allow_html=True)
