import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import datetime
import os

# --- Load TFLite Model ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="covdetect_model_float16.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

# Predict Function 
def predict_tflite(image):
    image = image.convert('RGB') 
    image = image.resize((180, 180))
    image = image.resize((180, 180))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(output)
    confidence = float(np.max(output))
    return predicted_class, confidence

# UI: Title & Intro 
st.title("CovDetect: Smart Disease Diagnosis")
st.write("Upload a an Xray image and fill patient info to predict disease and generate downloadable result.")

#  User Inputs 
name = st.text_input("👤 Patient Name")
age = st.number_input("📅 Age", min_value=1, max_value=120)
location = st.text_input("📍 Location")
image = st.file_uploader("🖼️ Upload Xray Image", type=['jpg', 'png', 'jpeg'])

# Predict Button 
if st.button("🔍 Diagnose"):
    if not all([name, age, location, image]):
        st.warning("Please fill in all fields and upload an image.")
    else:
        img = Image.open(image)
        predicted_class, confidence = predict_tflite(img)

        # Example class names (replace with your actual classes)
        class_names = ['Healthy', 'Leaf Spot', 'Rust', 'Blight']
        diagnosis = class_names[predicted_class]

        st.success(f"🩺 Diagnosis: **{diagnosis}** (Confidence: {confidence:.2%})")

        # Save to results.csv 
        result = {
            "Name": name,
            "Age": age,
            "Location": location,
            "Date": datetime.datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            "Diagnosis": diagnosis,
            "Confidence": f"{confidence:.2%}"
        }

        if os.path.exists("results.csv"):
            df = pd.read_csv("results.csv")
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        else:
            df = pd.DataFrame([result])
        
        df.to_csv("results.csv", index=False)
        st.download_button("📥 Download Results CSV", data=df.to_csv(index=False), file_name="results.csv", mime='text/csv')

# --- Show Results Table
if os.path.exists("results.csv"):
    with st.expander("📊 View Previous Results"):
        st.dataframe(pd.read_csv("results.csv"))
