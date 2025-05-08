import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# --- Load model once ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pcos_classifier_mobilenetv2.keras")
model = load_model()

# --- Initialize page state ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- CSS for sidebar buttons ---
st.markdown("""<style>
[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    text-align: left;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background-color: transparent;
    border: none;
    border-radius: 4px;
    transition: background-color 0.2s ease;
    font-size: 1.25rem;
    font-weight: 500;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background-color: #e0e0e0;
}
[data-testid="stSidebar"] .stButton>button:focus {
    background-color: #c0c0c0;
}
</style>""", unsafe_allow_html=True)

# --- Sidebar navigation ---
with st.sidebar:
    if st.button("Home", key="btn_home"):
        st.session_state.page = "Home"
    if st.button("Detect PCOS", key="btn_detect"):
        st.session_state.page = "Detect PCOS"

# --- Render pages ---
if st.session_state.page == "Home":
    st.markdown("""<div style="display:flex; justify-content:center; margin-top:50px;">
<div style="max-width:600px; width:100%;">

  <div style="text-align:center;">
    <h1>PCOS Ultrasound Detector</h1>
    <p><strong>Disclaimer:</strong> For research/educational use only. Not a substitute for professional medical advice.</p>
  </div>

  <div style="text-align:center; margin-top:2rem;">
    <h4>Instructions</h4>
    <ol style="display:inline-block; text-align:left; margin:0 auto; max-width:400px;">
      <li>Click “Detect PCOS” in the sidebar.</li>
      <li>Upload an ultrasound image of an Ovary with/without PCOS (JPG).</li>
      <li>Click <strong>Predict</strong> to view the result.</li>
    </ol>
  </div>

  <div style="margin-top:2rem; text-align:left; max-width:600px; margin:auto;">
    <h4>Technical Details</h4>
    <p>
      This classifier uses Transfer Learning with a MobileNetV2 backbone pre-trained on ImageNet.  
      Input ultrasound images are resized to 224×224 pixels, preprocessed with MobileNetV2’s 
      preprocessing function, and fed through the frozen convolutional base.  
      A GlobalAveragePooling layer and Dropout (rate 0.2) are applied before the final Dense layer 
      with sigmoid activation. Training uses the Adam optimizer (lr=1e-4), binary crossentropy loss, 
      ReduceLROnPlateau and EarlyStopping callbacks.
    </p>
  </div>

</div>
</div>""", unsafe_allow_html=True)

else:
    st.markdown("""<div style="display:flex; justify-content:center; margin-top:50px;">
<div style="max-width:600px; width:100%; text-align:center;">
  <h2>Detect PCOS</h2>
</div>
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], key="uploader")
    if uploaded:
        st.image(uploaded, width=300, caption="🔍 Preview")
        if st.button("Predict", key="btn_predict"):
            img = image.load_img(uploaded, target_size=(224,224), color_mode="rgb")
            arr = image.img_to_array(img)[None, ...]
            score = float(model.predict(arr)[0][0])
            if score < 0.5:
                st.error(f"🚨 PCOS detected")
            else:
                st.success(f"✅ PCOS not detected")
