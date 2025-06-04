import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image

# 📌 Download model from Google Drive if not present locally
def download_model():
    model_path = "trained_model.keras"
    if not os.path.exists(model_path):
        file_id = "10kq0xS3WKsaz1YHiQ64Rjn2Q-xhHk4kt"
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write("⏳ Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    return model_path

# 🔽 Load model with caching for faster inference
@st.cache_resource
def load_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path)

model = load_model()

# 🌿 Function to preprocess image and predict disease
def model_predict(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 224, 224, 3)
        
        prediction = model.predict(img)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return result_index, confidence
    except Exception as e:
        return None, str(e)

# 📌 Sidebar menu
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# 📌 Display logo from GitHub instead of local path
img_url = "https://raw.githubusercontent.com/sree260632/ESR-PLANT-SCAN/main/logo_app.png"
try:
    img = Image.open(img_url)
    st.image(img)
except Exception as e:
    st.warning(f"⚠️ Logo image not found: {e}")

# 🌿 Class labels for plant diseases
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# 📌 Home Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>🌾 Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True
    )

# 📌 Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("🌿 Plant Disease Recognition")
    test_image = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())
        
        if st.button("Show Image"):
            st.image(test_image, width=400, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔍 **Model is analyzing...**")

            result_index, confidence = model_predict(save_path)

            if result_index is not None:
                st.success(f"✅ Model predicts: **{class_name[result_index]}**")
                st.info(f"📊 Confidence Score: **{confidence:.2f}**")
            else:
                st.error(f"⚠️ Prediction failed: {confidence}")

            # Clean up temporary files
            os.remove(save_path)

