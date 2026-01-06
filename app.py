"""
Kidney Stone Detection using Vision Transformers
A Streamlit application for medical image analysis
"""

from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import streamlit as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setting page layout - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Kidney Stone Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .detection-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border: 1px solid #4caf50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / 'images'
MODEL_DIR = ROOT / 'weights'
DEFAULT_IMAGE = IMAGES_DIR / 'STONE- (15).jpg'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Main page heading
st.markdown('<h1 class="main-header">🏥 Kidney Stone Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis using Vision Transformers</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.image("https://img.icons8.com/fluency/96/kidney.png", width=80)
st.sidebar.title("⚙️ Configuration")

# Model loading with error handling
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with comprehensive error handling"""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        return model, None
    except ImportError as e:
        return None, f"ultralytics package not installed: {e}"
    except Exception as e:
        return None, f"Error loading model: {e}"

# Check model file exists
model_path = DETECTION_MODEL
model = None
model_error = None

if not model_path.exists():
    model_error = f"Model file not found at: {model_path}"
    st.sidebar.error(f"❌ {model_error}")
else:
    model, model_error = load_model(model_path)
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.error(f"❌ {model_error}")

# Confidence threshold
st.sidebar.markdown("### Detection Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Higher values = more confident detections only"
)

# Image Upload Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Image Upload")
source_img = st.sidebar.file_uploader(
    "Choose an image...",
    type=("jpg", "jpeg", "png", 'bmp', 'webp'),
    help="Upload a CT scan or ultrasound image of kidneys"
)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Input Image")
    if source_img is None:
        # Show default image
        if DEFAULT_IMAGE.exists():
            default_image = Image.open(DEFAULT_IMAGE)
            st.image(default_image, caption="Sample Image - Upload your own image", use_container_width=True)
        else:
            st.info("👆 Please upload an image using the sidebar")
            st.markdown("""
            <div style="border: 2px dashed #ccc; padding: 50px; text-align: center; border-radius: 10px;">
                <h3>📷 No Image Uploaded</h3>
                <p>Upload a kidney CT scan or ultrasound image to begin detection</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        uploaded_image = Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        # Display image info
        st.markdown(f"""
        <div class="result-box">
            <strong>Image Info:</strong><br>
            • Size: {uploaded_image.size[0]} x {uploaded_image.size[1]} pixels<br>
            • Format: {uploaded_image.format or 'Unknown'}<br>
            • Mode: {uploaded_image.mode}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🔍 Detection Results")
    
    if source_img is None:
        # Show default detected image or placeholder
        if DEFAULT_IMAGE.exists():
            detected_image = Image.open(DEFAULT_IMAGE)
            st.image(detected_image, caption='Sample Detection Result', use_container_width=True)
        else:
            st.markdown("""
            <div style="border: 2px dashed #ccc; padding: 50px; text-align: center; border-radius: 10px;">
                <h3>🎯 Detection Output</h3>
                <p>Results will appear here after detection</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Detection button
        if st.button('🔬 Detect Kidney Stones', type='primary', use_container_width=True):
            if model is not None:
                with st.spinner('🔄 Analyzing image...'):
                    try:
                        # Run detection
                        uploaded_image = Image.open(source_img)
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        
                        # Display results
                        st.image(res_plotted, caption='Detection Result', use_container_width=True)
                        
                        # Detection statistics
                        num_detections = len(boxes)
                        
                        if num_detections > 0:
                            st.success(f"✅ Found {num_detections} potential kidney stone(s)")
                            
                            # Show detailed results
                            with st.expander("📊 Detailed Detection Results", expanded=True):
                                for i, box in enumerate(boxes):
                                    conf_score = float(box.conf[0])
                                    cls_id = int(box.cls[0])
                                    
                                    # Get class name if available
                                    class_name = model.names.get(cls_id, f"Class {cls_id}") if hasattr(model, 'names') else f"Class {cls_id}"
                                    
                                    st.markdown(f"""
                                    <div class="detection-info">
                                        <strong>Detection {i+1}:</strong> {class_name}<br>
                                        <strong>Confidence:</strong> {conf_score:.2%}<br>
                                        <strong>Bounding Box:</strong> {box.xyxy[0].tolist()}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Export option
                            st.download_button(
                                label="📥 Download Detection Results",
                                data=str(res[0].boxes.data.tolist()),
                                file_name="detection_results.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("⚠️ No kidney stones detected in this image")
                            st.info("Try adjusting the confidence threshold or upload a different image")
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
            else:
                st.error(f"Model not loaded: {model_error}")
                st.info("Please check the installation requirements.")
        else:
            st.info("👆 Click the 'Detect Kidney Stones' button to analyze the image")

# Footer with information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <h4>About This Application</h4>
    <p>This application uses Vision Transformers with LSA and Shift Packet Tokenization 
    for accurate kidney stone detection in medical images.</p>
    <p><strong>Note:</strong> This tool is for educational/research purposes only. 
    Always consult healthcare professionals for medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Instructions
1. Upload a kidney CT scan or ultrasound image
2. Adjust confidence threshold if needed
3. Click 'Detect Kidney Stones'
4. View and download results

### 🔬 Model Info
- Architecture: YOLOv8 + Vision Transformers
- Input: Medical Images
- Output: Bounding boxes with confidence scores
""")
