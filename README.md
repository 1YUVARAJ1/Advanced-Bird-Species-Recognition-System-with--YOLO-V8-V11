# 🦜 Advanced Bird Species Recognition System

A high-performance, academic-grade Computer Vision application combining **YOLO (You Only Look Once)** for dynamic object detection and **EfficientNet-B3** for fine-grained image classification on the CUB-200-2011 dataset.

## 🌟 Architecture Overview: The Hybrid Pipeline

Identifying bird species is notoriously difficult due to extreme similarities between species (inter-class visual overlap). To counteract this, our architecture decouples detection from classification:

1. **Detection Layer (YOLO):** Before classifying, analyzing the whole picture creates 'background noise' (trees, lakes, skies). YOLO acts as a powerful attention mechanism, drawing lightning-fast bounding boxes around the bird and cropping it completely from the background. 
2. **Classification Layer (EfficientNet-B3):** The cleanly cropped bird image is resized to `224x224` and parsed through EfficientNet-B3 to decipher complex feather shapes, beak sizes, and colors out of 200 distinct species. 
3. **Metadata Enrichment:** The prediction is linked directly to a Wikipedia/GBIF dataset library, populating educational stats such as Conservation Status, Diet, and Lifespan in the UI.

## ⚡ The Advanced Benchmarking Platform

Run `app_advanced.py` to launch our **Academic Benchmarking Interface**:
* **Dynamic Model Switching:** Instantly swap between YOLO variants (`v8n`, `v8x`, `11n`, `11x`) inside the UI Sidebar to calculate how deep-learning parameter counts affect FPS and precision. 
* **Model Cache Registry:** When models are switched, they are intelligently cached in RAM, skipping extreme read/write cycles on traditional hard drives. 
* **Hardware Compute Toggles:** If PyTorch detects NVIDIA CUDA, you can physically hot-swap neural computation paths directly from `CPU` to `GPU` in real-time.
* **Execution Metrics:** Every processed frame prints an execution breakdown, measuring exact Inference Millisecond times between object detection and classification.

---

## 🚀 Setup & Installation (For Friends & Reviewers)

**1. Clone the Repository:**
```bash
git clone https://github.com/1YUVARAJ1/Advanced-Bird-Species-Recognition-System-with--YOLO-V8-V11.git
cd Advanced-Bird-Species-Recognition-System-with--YOLO-V8-V11
```

**2. Create a Virtual Environment (Optional but Recommended):**
```bash
python -m venv bird_env
# Windows:
.\bird_env\Scripts\activate
# Mac/Linux:
source bird_env/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Benchmarking Application:**
```bash
python run_app.py
```
*(The UI will automatically open in your default browser at `http://localhost:8501`)*

---
**Note:** The `/src/models/weights/efficientnet_b3_best.pth` file is already cached within this repository. No extra training is required! 
