import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from src.models.inference_advanced import BirdRecognitionPipeline

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
    
    /* Global Text & Table Scaling */
    p, li {
        font-size: 1.1rem !important;
    }
    table {
        font-size: 1.2rem !important;
        width: 100%;
    }
    th, td {
        padding: 12px 16px !important;
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
    yolo_choice = "yolo11n.pt"
    device_choice = "cpu"
    
    def __init__(self):
        self.last_results = None
        self.last_update_time = 0

    def transform(self, frame):
        img_np = frame.to_ndarray(format="bgr24")
        
        start_time = time.time()
        
        # 1. Very basic downsample for YOLO speed if needed
        # 2. Run detection
        det_start = time.time()
        bbox, yolo_conf = pipeline.detect_bird(img_np, self.yolo_choice, self.device_choice)
        det_time = time.time() - det_start
        
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
                        
                        class_start = time.time()
                        top_preds = pipeline.predict_species(cropped_pil)
                        class_time = time.time() - class_start
                        
                        best_pred = top_preds[0]
                        meta = pipeline.get_metadata(best_pred['species'])
                        
                        total_time = time.time() - start_time
                        
                        self.last_results = {
                            "top_predictions": top_preds,
                            "best_prediction": best_pred,
                            "metadata": meta,
                            "metrics": {
                                "detection_time_ms": det_time * 1000,
                                "classification_time_ms": class_time * 1000,
                                "total_time_ms": total_time * 1000
                            }
                        }
                        self.last_update_time = current_time
                except Exception as e:
                    pass
            
        return img_np

def main():
    st.title("🦜 Advanced Bird Species Recognition System")
    st.markdown("### Dynamic YOLO + EfficientNetB3 Hybrid Architecture")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["🏠 Home", "📷 Image Upload", "📹 Live Camera (WIP)", "📊 Model Info"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Benchmarking Settings")
    
    import torch
    cuda_available = torch.cuda.is_available()
    devices = ["cpu", "cuda"]
    
    def format_device(d):
        if d == "cpu": return "CPU"
        if d == "cuda": return "GPU (CUDA)" if cuda_available else "GPU (CUDA - Missing PyTorch Drivers)"
        
    device_choice = st.sidebar.selectbox("Compute Device", devices, format_func=format_device)
    
    if device_choice == "cuda" and not cuda_available:
        st.sidebar.warning("⚠️ GPU selected, but PyTorch cannot detect NVIDIA CUDA drivers. Falling back to CPU processing.")
        device_choice = "cpu"
        
    pipeline.set_device(device_choice)
    
    yolo_choice = st.sidebar.selectbox(
        "Detection Model", 
        ["yolov8n.pt", "yolov8x.pt", "yolo11n.pt", "yolo11x.pt"],
        index=2 # Defaults to yolo11n.pt automatically
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset:** CUB-200-2011")
    st.sidebar.markdown("**Classification:** EfficientNet-B3 (Target: ~95% acc)")
    
    if app_mode == "🏠 Home":
        st.markdown("""
        Welcome to the **Advanced Bird Species Recognition System**.
        
        This application uses a state-of-the-art hybrid deep learning pipeline:
        1. **YOLO Detection** identifies the bird region to ignore background noise.
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
                results = pipeline.run_pipeline(tmp.name, yolo_model_name=yolo_choice, device_name=device_choice)
            
            try:
                os.remove(tmp.name)
            except Exception:
                pass
            
            with col1:
                st.markdown('<div class="stCard">', unsafe_allow_html=True)
                
                # Draw Box if detected
                if results['has_bird_detected']:
                    display_img = draw_bbox(image, results['yolo_bbox'], results['yolo_conf'])
                    st.image(display_img, caption=f"{yolo_choice.upper()} Detection", use_container_width=True)
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
                    
                    st.markdown("### 📊 Species Profile")
                    st.markdown(f"""
| Attribute | Information |
| :--- | :--- |
| **Common Name** | {species} |
| **Scientific Name** | <i>{scientific_name}</i> |
| **Family** | {family} |
| **Conservation** | {status_color(iucn)} {iucn} |
| **Habitat** | {habitat} |
| **Diet** | {diet} |
| **Lifespan** | {lifespan} |
""", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown("#### ⚡ Performance Benchmarks")
                    if 'metrics' in results:
                        m = results['metrics']
                        metrics_table = f"""
| Phase | Execution Time (ms) |
| :--- | :--- |
| **YOLO Detection** | {m['detection_time_ms']:.1f} |
| **EfficientNet Classification** | {m['classification_time_ms']:.1f} |
| **Total Inference Time** | **{m['total_time_ms']:.1f}** |
"""
                        st.markdown(metrics_table)
                    
                    st.markdown("---")
                    st.markdown("#### Top 5 Predictions:")
                    pred_table = "| Rank | Species | Confidence |\n| :--- | :--- | :--- |\n"
                    for i, pred in enumerate(results['top_predictions']):
                        s_name = pred['species'].replace('_', ' ')
                        p_val = pred['prob'] * 100
                        pred_table += f"| {i+1} | {s_name} | {p_val:.1f}% |\n"
                    st.markdown(pred_table)
                
    elif app_mode == "📹 Live Camera (WIP)":
        st.header("Real-Time Webcam Detection")
        st.info(f"This view runs **{yolo_choice.upper()}** directly on your webcam feed to track birds in real-time. It runs EfficientNet every 1 second to update classification stats without lagging the video.")
        
        ctx = webrtc_streamer(
            key="bird-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        if ctx.video_processor:
            ctx.video_processor.yolo_choice = yolo_choice
            ctx.video_processor.device_choice = device_choice
            
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
                            
                            st.markdown("### 📊 Species Profile")
                            st.markdown(f"""
| Attribute | Information |
| :--- | :--- |
| **Common Name** | {species} |
| **Scientific Name** | <i>{scientific_name}</i> |
| **Family** | {family} |
| **Conservation** | {status_color(iucn)} {iucn} |
| **Habitat** | {habitat} |
| **Diet** | {diet} |
| **Lifespan** | {lifespan} |
""", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            st.markdown("#### ⚡ Performance Benchmarks")
                            if 'metrics' in res:
                                m = res['metrics']
                                metrics_table = f"""
| Phase | Execution Time (ms) |
| :--- | :--- |
| **YOLO Detection** | {m['detection_time_ms']:.1f} |
| **EfficientNet Classification** | {m['classification_time_ms']:.1f} |
| **Total Inference Time** | **{m['total_time_ms']:.1f}** |
"""
                                st.markdown(metrics_table)
                            
                            st.markdown("---")
                            st.markdown("#### Top 5 Predictions:")
                            pred_table = "| Rank | Species | Confidence |\n| :--- | :--- | :--- |\n"
                            for i, pred in enumerate(res['top_predictions']):
                                s_name = pred['species'].replace('_', ' ')
                                p_val = pred['prob'] * 100
                                pred_table += f"| {i+1} | {s_name} | {p_val:.1f}% |\n"
                            st.markdown(pred_table)
                
                time.sleep(0.5)
        
    elif app_mode == "📊 Model Info":
        st.header("📊 Project Architecture & Real-Time Benchmarking")
        
        st.markdown("""
        ### 🌟 Project Overview
        The **Advanced Bird Species Recognition System** is a professional-grade computer vision application designed to tackle the complex problem of **fine-grained image classification**. Identifying bird species is notoriously difficult due to striking similarities between different species (inter-class similarity) and significant variations within the same species due to age, gender, and season (intra-class variance).

        To solve this, the application employs a **Hybrid Deep Learning Pipeline** that decouples object detection from species classification, resulting in a highly robust and accurate system.

        ---

        ### 🧠 The Hybrid Pipeline Explained

        #### 1. Detection Layer: YOLO (You Only Look Once)
        Before classification occurs, the image is passed through a state-of-the-art YOLO object detection model. 
        * **Purpose:** To locate the bird within the frame and draw a bounding box around it.
        * **Benefit:** By cropping the image exactly to the bird's coordinates, we eliminate background noise (trees, sky, water) that could confuse the classifier. This acts as a highly effective attention mechanism.
        * **Dynamic Selection:** The system supports dynamic hot-swapping between lightweight edge models (`YOLOv8n`, `YOLO11n`) for maximum FPS, and highly accurate models (`YOLOv8x`, `YOLO11x`) for maximum precision.

        #### 2. Classification Layer: EfficientNet-B3
        Once the bird is isolated, the tightly cropped image is resized to `224x224` and passed to an EfficientNet-B3 Convolutional Neural Network.
        * **Purpose:** To extract intricate, fine-grained visual features (beak shape, feather patterns, eye color) and classify the bird into one of 200 possible species.
        * **Why EfficientNet?** EfficientNet utilizes a compound scaling method that uniformly scales network width, depth, and resolution. This achieves state-of-the-art accuracy (`~95%` target on CUB-200) while remaining computationally efficient enough to run alongside YOLO.

        #### 3. Enrichment Layer: Biodiversity Metadata
        The raw species prediction is cross-referenced against a curated knowledge base (sourced from Wikipedia and GBIF APIs) to provide rich, educational context, including the bird's scientifically accepted binomial name, family, habitat, diet, and IUCN conservation status.

        ---

        ### ⚡ The Academic Benchmarking Platform
        This specific _advanced_ version of the application acts as a real-time benchmarking sandbox. 
        
        * **Lazy-Loading `MODEL_REGISTRY`:** Models are intelligently cached in RAM. Switching between YOLO models only accesses the disk once, enabling instant comparisons without crashing older CPUs.
        * **Hardware Compute Toggles:** PyTorch dynamically probes the system for NVIDIA CUDA availability. Users can instantly switch the neural network logic between the `CPU` and `GPU`, visualizing the dramatic difference in deep learning inference speeds.
        * **Execution Metrics:** Every frame processed calculates the exact millisecond execution time of both the Detection and Classification pipelines, rendering the performance tradeoff of speed vs. accuracy visible in the UI.

        ---

        ### 📁 Data Source
        **Caltech-UCSD Birds-200-2011 (CUB-200-2011)**
        * 11,788 precisely curated images.
        * 200 distinct bird species categorized by expert ornithologists.
        """)

if __name__ == "__main__":
    main()
