
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

from model import CorruptionCNN
CLASSES = [
    "clean",
    "gaussian_noise",
    "blur",
    "brightness",
    "contrast",
    "jpeg_artifacts",
    "mixed"
]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CORRUPTION.AI | Image Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
def load_css():
    css_path = Path("style.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )

load_css()

# ---------------- MODEL LOADING ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/corruption_cnn.pth")

@st.cache_resource
def load_model():
    model = CorruptionCNN().to(DEVICE)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.convert("RGB").resize((32, 32))
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    # CIFAR-10 normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(DEVICE)

# ---------------- HEADER ----------------
st.markdown("""
<div class="topbar">
    <div class="brand">⚡ CORRUPTION<span>.AI</span></div>
    <div class="status"><span class="pulse"></span> AI SYSTEM ONLINE</div>
</div>

<div class="hero">
    <div class="eyebrow">NEXT GENERATION IMAGE INTELLIGENCE</div>
    <h1>SEE BEYOND THE <span>PIXELS.</span></h1>
    <p>Advanced deep learning for image corruption detection and robustness analysis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- METRIC CARDS ----------------
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric"><small>ENGINE</small><h3>CNN</h3><p>Deep learning model</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric"><small>INPUT</small><h3>32 × 32</h3><p>Model image resolution</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric"><small>COMPUTE</small><h3>{str(DEVICE).upper()}</h3><p>Active processing device</p></div>', unsafe_allow_html=True)

st.markdown("<div class='section-title'>01 / IMAGE ANALYSIS</div>", unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="panel-heading">UPLOAD YOUR IMAGE</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an image here or browse",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload an image for corruption classification."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="INPUT IMAGE", use_container_width=True)
        st.caption(f"Original dimensions: {image.width} × {image.height}")
    else:
        st.markdown("""
        <div class="upload-hint">
            <div class="upload-icon">⬆</div>
            <h3>DROP IMAGE HERE</h3>
            <p>PNG · JPG · JPEG · WEBP</p>
            <small>Waiting for image input...</small>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-heading">AI DIAGNOSTICS</div>', unsafe_allow_html=True)

    if uploaded_file:
        if st.button("⚡ ANALYZE IMAGE", use_container_width=True):
            if not MODEL_PATH.exists():
                st.error("Model file not found. Check models/corruption_cnn.pth.")
            else:
                try:
                    with st.spinner("Neural network processing image..."):
                        model = load_model()
                        input_tensor = preprocess_image(image)

                        with torch.no_grad():
                            logits = model(input_tensor)
                            probabilities = F.softmax(logits, dim=1)[0]

                        if len(CLASSES) != probabilities.numel():
                            st.error("Class count mismatch between model and CLASSES.")
                        else:
                            predicted_index = int(torch.argmax(probabilities).item())
                            predicted_class = CLASSES[predicted_index]
                            confidence = probabilities[predicted_index].item() * 100

                            st.session_state["analysis"] = {
                                "prediction": predicted_class,
                                "confidence": confidence,
                                "probabilities": probabilities.cpu().tolist()
                            }
                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    result = st.session_state.get("analysis")

    if result:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">DETECTED CORRUPTION</div>
            <h2>{result['prediction']}</h2>
            <div class="confidence">{result['confidence']:.2f}<span>%</span></div>
            <p>MODEL CONFIDENCE</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### CLASS PROBABILITY DISTRIBUTION")

        for class_name, probability in zip(CLASSES, result["probabilities"]):
            percentage = probability * 100
            st.markdown(
                f"<div class='prob-label'><span>{class_name}</span><b>{percentage:.2f}%</b></div>",
                unsafe_allow_html=True
            )
            st.progress(float(probability))
    else:
        st.markdown("""
        <div class="waiting-card">
            <div class="waiting-icon">◈</div>
            <h3>AWAITING INPUT</h3>
            <p>Upload an image and start the AI analysis to reveal the prediction.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <div>⚡ CORRUPTION.AI</div>
    <p>IMAGE ROBUSTNESS INTELLIGENCE · DEEP LEARNING PROJECT</p>
</div>
""", unsafe_allow_html=True)
