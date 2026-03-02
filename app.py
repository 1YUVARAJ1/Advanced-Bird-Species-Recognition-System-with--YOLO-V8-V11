import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from src.models.inference import BirdRecognitionPipeline

st.set_page_config(
    page_title="Advanced Bird Species Recognition System",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize the pipeline once
@st.cache_resource
def load_pipeline():
    return BirdRecognitionPipeline()

pipeline = load_pipeline()

# --- Custom CSS for Dark Mode and Premium Feel ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E2E8F0;
        font-weight: 700;
    }
    
    /* Accent color for highlights */
    .highlight {
        color: #38BDF8;
        font-weight: 600;
    }
    
    /* Cards for Images and Results */
    .stCard {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
        transition: transform 0.2s ease-in-out;
    }
    .stCard:hover {
        transform: translateY(-2px);
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background-color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)

def status_color(status):
    status_lower = status.lower()
    if "least concern" in status_lower:
        return "🟢"
    elif "near threatened" in status_lower:
        return "🟡"
    elif "endangered" in status_lower or "vulnerable" in status_lower:
        return "🔴"
    else:
        return "⚪"

def draw_bbox(image_pil, bbox, confidence):
    """ Draw bounding box on PIL image using OpenCV """
    if isinstance(image_pil, np.ndarray):
        img_cv = image_pil
    else:
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 193, 7), 3) # Amber color for bbox
    
    text = f"Bird: {confidence*100:.1f}%"
    cv2.putText(img_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 193, 7), 2)
    
    if isinstance(image_pil, np.ndarray):
        return img_cv
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_results = None
        self.last_update_time = 0

    def transform(self, frame):
        img_np = frame.to_ndarray(format="bgr24")
        
        # 1. Very basic downsample for YOLO speed if needed
        # 2. Run detection
        bbox, yolo_conf = pipeline.detect_bird(img_np)
        
        # 3. Draw Box
        if bbox is not None:
            img_np = draw_bbox(img_np, bbox, yolo_conf)
            
            # Rate limit EfficientNet to 1 frame per second to save FPS
            current_time = time.time()
            if current_time - self.last_update_time > 1.0:
                try:
                    # Crop logic for EfficientNet
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = img_np.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped = img_np[y1:y2, x1:x2]
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        
                        top_preds = pipeline.predict_species(cropped_pil)
                        best_pred = top_preds[0]
                        meta = pipeline.get_metadata(best_pred['species'])
                        
                        self.last_results = {
                            "top_predictions": top_preds,
                            "best_prediction": best_pred,
                            "metadata": meta
                        }
                        self.last_update_time = current_time
                except Exception as e:
                    pass
            
        return img_np

def main():
    st.title("🦜 Advanced Bird Species Recognition System")
    st.markdown("### YOLOv8 + EfficientNetB3 Hybrid Architecture")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["🏠 Home", "📷 Image Upload", "📹 Live Camera (WIP)", "📊 Model Info"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset:** CUB-200-2011")
    st.sidebar.markdown("**Detection:** YOLOv8n")
    st.sidebar.markdown("**Classification:** EfficientNet-B3 (Target: ~95% acc)")
    
    if app_mode == "🏠 Home":
        st.markdown("""
        Welcome to the **Advanced Bird Species Recognition System**.
        
        This application uses a state-of-the-art hybrid deep learning pipeline:
        1. **YOLOv8** detects the bird region to ignore background noise.
        2. **EfficientNetB3** classifies the precise bird species.
        3. **Biodiversity Metadata** enriches the prediction with habitat and IUCN status.
        
        👈 **Select an option from the sidebar to get started.**
        """)
        
    elif app_mode == "📷 Image Upload":
        st.header("Image Upload & Recognition")
        uploaded_file = st.file_uploader("Upload a bird image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            # Read image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Save temporarily for pipeline closing it first so Windows can access
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.close()
            image.save(tmp.name)
            
            with st.spinner("Analyzing image..."):
                results = pipeline.run_pipeline(tmp.name)
            
            try:
                os.remove(tmp.name)
            except Exception:
                pass
            
            with col1:
                st.markdown('<div class="stCard">', unsafe_allow_html=True)
                
                # Draw Box if detected
                if results['has_bird_detected']:
                    display_img = draw_bbox(image, results['yolo_bbox'], results['yolo_conf'])
                    st.image(display_img, caption="YOLOv8 Detection", use_container_width=True)
                else:
                    st.image(image, caption="No High-Confidence Bird Box Detected (Using full image)", use_container_width=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                if 'error' in results:
                    st.error(results['error'])
                else:
                    best_pred = results['best_prediction']
                    species = best_pred['species'].replace('_', ' ')
                    prob = best_pred['prob'] * 100
                    
                    st.markdown("## 🕊️ Bird Species Identified")
                    st.success(f"{species}")
                    
                    st.markdown(f"🎯 **Confidence:** {prob:.2f}%")
                    st.markdown("---")
                    
                    meta = results['metadata']
                    scientific_name = meta.get('scientific_name', 'Unknown')
                    family = meta.get('family', 'Unknown')
                    habitat = meta.get('habitat', 'Unknown')
                    diet = meta.get('diet', 'Unknown')
                    lifespan = meta.get('lifespan', 'Unknown')
                    iucn = meta.get('iucn_status', 'Not Evaluated')
                    
                    st.markdown(f"**Common Name:** {species}")
                    st.markdown(f"**Scientific Name:** {scientific_name}")
                    st.markdown(f"**Family:** {family}")
                    st.markdown(f"**Conservation:** {status_color(iucn)} {iucn}")
                    st.markdown("---")
                    
                    st.markdown(f"🌍 **Habitat:** {habitat}")
                    st.markdown(f"🍽️ **Diet:** {diet}")
                    st.markdown(f"⏳ **Lifespan:** {lifespan}")
                    
                    st.markdown("---")
                    st.markdown("#### Top 5 Predictions:")
                    for i, pred in enumerate(results['top_predictions']):
                        s_name = pred['species'].replace('_', ' ')
                        p_val = pred['prob'] * 100
                        st.markdown(f"{i+1}. {s_name} – {p_val:.1f}%")
                
    elif app_mode == "📹 Live Camera (WIP)":
        st.header("Real-Time Webcam Detection")
        st.info("This view runs YOLOv8 directly on your webcam feed to track birds in real-time. It runs EfficientNet every 1 second to update classification stats without lagging the video.")
        
        ctx = webrtc_streamer(
            key="bird-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        if ctx.state.playing:
            st.markdown("---")
            results_placeholder = st.empty()
            
            while True:
                if ctx.video_processor and hasattr(ctx.video_processor, 'last_results') and ctx.video_processor.last_results:
                    res = ctx.video_processor.last_results
                    with results_placeholder.container():
                        col1, col2 = st.columns([1, 1])
                        
                        best_pred = res['best_prediction']
                        species = best_pred['species'].replace('_', ' ')
                        prob = best_pred['prob'] * 100
                        
                        with col1:
                            st.markdown("## 🕊️ Bird Species Identified")
                            st.success(f"{species}")
                            
                            st.markdown(f"🎯 **Confidence:** {prob:.2f}%")
                            st.markdown("---")
                            
                            meta = res['metadata']
                            scientific_name = meta.get('scientific_name', 'Unknown')
                            family = meta.get('family', 'Unknown')
                            habitat = meta.get('habitat', 'Unknown')
                            diet = meta.get('diet', 'Unknown')
                            lifespan = meta.get('lifespan', 'Unknown')
                            iucn = meta.get('iucn_status', 'Not Evaluated')
                            
                            st.markdown(f"**Common Name:** {species}")
                            st.markdown(f"**Scientific Name:** {scientific_name}")
                            st.markdown(f"**Family:** {family}")
                            st.markdown(f"**Conservation:** {status_color(iucn)} {iucn}")
                            st.markdown("---")
                            
                            st.markdown(f"🌍 **Habitat:** {habitat}")
                            st.markdown(f"🍽️ **Diet:** {diet}")
                            st.markdown(f"⏳ **Lifespan:** {lifespan}")
                            
                            st.markdown("---")
                            st.markdown("#### Top 5 Predictions:")
                            for i, pred in enumerate(res['top_predictions']):
                                s_name = pred['species'].replace('_', ' ')
                                p_val = pred['prob'] * 100
                                st.markdown(f"{i+1}. {s_name} – {p_val:.1f}%")
                
                time.sleep(0.5)
        
    elif app_mode == "📊 Model Info":
        st.header("Project Architecture")
        st.markdown("""
        **Data Source:** Caltech-UCSD Birds-200-2011 (CUB-200-2011)
        - 11,788 images across 200 bird species.
        - Extremely fine-grained classification.
        
        **Pipeline Steps:**
        1. **YOLOv8** identifies the coordinates of the bird in the image and isolates it.
        2. The cropped image box is resized to `224x224`.
        3. **EfficientNet-B3** (pre-trained on ImageNet1k) determines the species probabilities.
        """)

if __name__ == "__main__":
    main()
