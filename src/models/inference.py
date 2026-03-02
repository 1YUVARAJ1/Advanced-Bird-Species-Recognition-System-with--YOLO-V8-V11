import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import cv2
import requests

class BirdRecognitionPipeline:
    def __init__(self, efficientnet_weights='src/models/weights/efficientnet_b3_best.pth', metadata_csv='data/processed/bird_metadata.csv'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv8 for generic bird detection
        print("Loading YOLOv8...")
        self.detector = YOLO('yolov8n.pt') # Will download automatically if not present
        
        # Load EfficientNetB3 for classification
        print("Loading EfficientNetB3 classification model...")
        self.num_classes = 200
        self.classifier = models.efficientnet_b3(weights=None)
        num_ftrs = self.classifier.classifier[1].in_features
        self.classifier.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # Try loading weights if they exist
        if os.path.exists(efficientnet_weights):
            try:
                self.classifier.load_state_dict(torch.load(efficientnet_weights, map_location=self.device))
                print(f"Loaded weights from {efficientnet_weights}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        else:
            print(f"Warning: {efficientnet_weights} not found. Using untrained weights.")
            
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        # Transforms for EfficientNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load class names directly from folder structure
        self.meta_cache = {}
        
        # Load metadata and class names
        self.metadata = pd.read_csv(metadata_csv)
        
        # CRITICAL: PyTorch ImageFolder sorts classes alphabetically by folder name
        # We must align our class_names list to match this exact pythonic sorting
        data_dir = 'data/processed/CUB_200_2011_cropped/train'
        if os.path.exists(data_dir):
            raw_target_folders = sorted(os.listdir(data_dir))
            # The folder names look like '001.Black_footed_Albatross', we need just the name to index metadata
            self.class_names = []
            for f in raw_target_folders:
                clean_name = f.split('.')[-1] if '.' in f else f
                self.class_names.append(clean_name.replace('_', ' '))
        else:
            # If folders aren't found, fallback to the CSV mapping
            self.class_names = self.metadata['species'].tolist()
        
    def detect_bird(self, image_path_or_array):
        """ Run YOLO detection and return highest confidence bird bbox """
        results = self.detector(image_path_or_array, classes=[14], verbose=False) # class 14 is 'bird' in COCO dataset
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None # No bird detected
            
        # Get highest confidence bird
        boxes = results[0].boxes
        best_box_idx = torch.argmax(boxes.conf).item()
        best_box = boxes.xyxy[best_box_idx].cpu().numpy()
        confidence = boxes.conf[best_box_idx].cpu().item()
        
        return best_box, confidence

    def predict_species(self, image_pil):
        """ Run EfficientNet on cropped image """
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(img_t)
            
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        results = []
        for i in range(5):
            idx = top5_idx[i].item()
            prob = top5_prob[i].item()
            species_name = self.class_names[idx] if idx < len(self.class_names) else f"Unknown Class {idx}"
            results.append({"species": species_name, "prob": prob, "idx": idx})
            
        return results
        
    def get_metadata(self, species_name):
        """ Retrieve static metadata from the expert CSV file """
        if species_name in self.meta_cache:
            return self.meta_cache[species_name]
            
        meta = {
            "scientific_name": "Unknown",
            "family": "Unknown",
            "habitat": "Information unavailable.",
            "diet": "Unknown",
            "lifespan": "Unknown",
            "iucn_status": "Not Evaluated"
        }
        
        try:
            if hasattr(self, 'metadata') and not self.metadata.empty:
                match = self.metadata[self.metadata['species'] == species_name]
                if not match.empty:
                    meta['scientific_name'] = match.iloc[0].get('scientific_name', 'Unknown')
                    meta['family'] = match.iloc[0].get('family', 'Unknown')
                    meta['habitat'] = match.iloc[0].get('habitat', 'Unknown')
                    meta['diet'] = match.iloc[0].get('diet', 'Unknown')
                    meta['lifespan'] = match.iloc[0].get('lifespan', 'Unknown')
                    meta['iucn_status'] = match.iloc[0].get('iucn_status', 'Not Evaluated')
        except Exception as e:
            print(f"Metadata Fetch Error for {species_name}: {e}")
            
        self.meta_cache[species_name] = meta
        return meta

    def run_pipeline(self, image_path):
        """ Run full end-to-end inference """
        try:
            pil_img = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}
            
        # 1. Detect
        bbox, yolo_conf = self.detect_bird(image_path)
        
        # 2. Crop
        if bbox is not None:
            # bbox formatting [x1, y1, x2, y2]
            cropped_img = pil_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            has_bird = True
        else:
            cropped_img = pil_img # Fallback to whole image
            has_bird = False
            bbox = None
            yolo_conf = 0.0
            
        # 3. Classify
        top_preds = self.predict_species(cropped_img)
        
        # 4. Enrich
        best_pred = top_preds[0]
        meta = self.get_metadata(best_pred['species'])
        
        return {
            "has_bird_detected": has_bird,
            "yolo_bbox": bbox.tolist() if bbox is not None else None,
            "yolo_conf": yolo_conf,
            "top_predictions": top_preds,
            "best_prediction": best_pred,
            "metadata": meta
        }

if __name__ == "__main__":
    # Test block
    print("Testing pipeline initialization...")
    pipeline = BirdRecognitionPipeline()
    print("Pipeline ready.")
