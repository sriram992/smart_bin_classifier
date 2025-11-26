import streamlit as st
import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
import cv2
from PIL import Image
import pickle
import torch
import clip
import tempfile
from pathlib import Path

# ==========================================
# CONFIGURATION & FILE HOSTING
# ==========================================
# INSTRUCTIONS:
# 1. Upload your 'demo_bin_classifier_best.h5' and 'asin_vocabulary.pkl' to Google Drive or Dropbox.
# 2. Get the DIRECT download links. 
#    - For Google Drive, use a tool like https://sites.google.com/site/gdocs2direct/ to convert the share link.
#    - For Dropbox, change 'dl=0' to 'dl=1' at the end of the link.
# 3. Paste them below.

MODEL_URL = "https://drive.google.com/file/d/1g5deJ3i0h1843BXCO-Kn-2xRvaNzEsCb/view?usp=sharing"  
VOCAB_URL = "https://drive.google.com/file/d/1ET4ap1LK-yjr891U-WVJZT0oT1gwascD/view?usp=sharing"

# Local paths where files will be saved/checked
MODEL_PATH = "demo_bin_classifier_best.h5"
VOCAB_PATH = "asin_vocabulary.pkl"

st.set_page_config(page_title="Amazon Bin Classifier", layout="wide")

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def download_file(url, filename):
    """Downloads a large file if it doesn't exist locally."""
    if os.path.exists(filename):
        return True
    
    if "YOUR_DIRECT_LINK" in url:
        return False # User hasn't set the link yet

    try:
        with st.spinner(f"Downloading {filename}... (This happens once)"):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error downloading {filename}: {e}")
        return False

# ==========================================
# CORE CLASSES (Refactored for Inference)
# ==========================================

class CLIPFeatureExtractor:
    def __init__(self, model_name='ViT-B/32', device=None):
        self.device = "cpu" # Force CPU for Streamlit Cloud stability
        st.info(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.feature_dim = 512

    def extract_image_features(self, image_pil):
        with torch.no_grad():
            image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features.cpu().numpy().flatten()
            norm = np.linalg.norm(features) + 1e-10
            features = features / norm
        return features

class AmazonBinDataProcessor:
    def __init__(self, vocab_path, img_size=(224, 224)):
        self.img_size = img_size
        self.load_vocabulary(vocab_path)
        
    def load_vocabulary(self, path):
        try:
            with open(path, 'rb') as f:
                vocab = pickle.load(f)
            self.asin_to_idx = vocab.get('asin_to_idx', {})
            self.idx_to_asin = vocab.get('idx_to_asin', {})
            self.asin_to_name = vocab.get('asin_to_name', {})
        except Exception as e:
            st.warning("Could not load vocabulary. Predictions will show IDs instead of names.")
            self.asin_to_idx = {}
            self.idx_to_asin = {}
            self.asin_to_name = {}

    def preprocess_image(self, image_pil):
        # Convert PIL to CV2 format (RGB)
        img = np.array(image_pil)
        
        # Resize
        img = cv2.resize(img, tuple(self.img_size))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        return img

def build_hybrid_model_architecture(num_classes, img_size=(224, 224), clip_dim=512):
    """Rebuilds the architecture to load weights into"""
    cnn_input = layers.Input(shape=(*img_size, 3), name='cnn_input')
    backbone = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=cnn_input)
    cnn_features = backbone.output
    cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
    cnn_features = layers.Dropout(0.3)(cnn_features)
    cnn_features = layers.Dense(512, activation='relu', name='cnn_dense')(cnn_features)
    cnn_features = layers.BatchNormalization()(cnn_features)

    clip_input = layers.Input(shape=(clip_dim,), name='clip_input')
    clip_features = layers.Dense(512, activation='relu', name='clip_dense')(clip_input)
    clip_features = layers.BatchNormalization()(clip_features)
    clip_features = layers.Dropout(0.2)(clip_features)
    
    fused = layers.Concatenate(name='fusion')([cnn_features, clip_features])
    fused = layers.Dense(768, activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.4)(fused)

    item_branch = layers.Dense(256, activation='relu')(fused)
    item_branch = layers.Dropout(0.3)(item_branch)
    item_output = layers.Dense(num_classes, activation='sigmoid', name='item_presence')(item_branch)

    count_branch = layers.Dense(128, activation='relu')(fused)
    count_branch = layers.Dropout(0.3)(count_branch)
    count_output = layers.Dense(1, activation='relu', name='item_count')(count_branch)

    quantity_branch = layers.Dense(128, activation='relu')(fused)
    quantity_branch = layers.Dropout(0.3)(quantity_branch)
    quantity_output = layers.Dense(1, activation='relu', name='total_quantity')(quantity_branch)

    model = Model(inputs=[cnn_input, clip_input], outputs=[item_output, count_output, quantity_output])
    return model

# ==========================================
# APP LOGIC
# ==========================================

@st.cache_resource
def load_resources():
    # 1. Check/Download Vocabulary
    has_vocab = download_file(VOCAB_URL, VOCAB_PATH)
    
    # 2. Check/Download Model
    has_model = download_file(MODEL_URL, MODEL_PATH)

    # 3. Load CLIP (Always needed)
    clip_extractor = CLIPFeatureExtractor()
    
    processor = None
    model = None
    
    if has_vocab:
        processor = AmazonBinDataProcessor(VOCAB_PATH)
    
    if has_model and processor:
        try:
            # We try to load the full model first
            model = keras.models.load_model(MODEL_PATH)
        except:
            st.warning("Direct load failed, rebuilding architecture and loading weights...")
            # Fallback: Rebuild arch and load weights
            num_classes = len(processor.asin_to_idx) if processor.asin_to_idx else 200 # Default fallback
            model = build_hybrid_model_architecture(num_classes)
            model.load_weights(MODEL_PATH)
            
    return clip_extractor, processor, model

st.title("📦 Amazon Bin Classifier (Hybrid CLIP+CNN)")

st.markdown("""
This app classifies items in Amazon bins using a hybrid EfficientNetV2 + CLIP architecture.
Upload an image of a bin to detect items, count, and quantity.
""")

# Load resources
clip_extractor, processor, model = load_resources()

# UI State handling for missing files
if model is None:
    st.error("⚠️ Model file not found or failed to load.")
    st.warning("""
    **Deployment Instructions:**
    1. Host your `.h5` and `.pkl` files on a public URL (Google Drive, Hugging Face, etc).
    2. Edit `app.py` lines 25-26 to include the direct download links.
    3. If running locally, simply place the files in the same folder as this script.
    """)
else:
    st.success("System Ready: Model and Vocabulary Loaded")

    uploaded_file = st.file_uploader("Choose a bin image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.subheader("Analysis")
            
            with st.spinner("Processing image..."):
                # 1. Preprocess Image (CNN Input)
                img_processed = processor.preprocess_image(image_pil)
                img_batch = np.expand_dims(img_processed, axis=0)
                
                # 2. Extract CLIP Features
                clip_features = clip_extractor.extract_image_features(image_pil)
                clip_batch = np.expand_dims(clip_features, axis=0)
                
                # 3. Predict
                predictions = model.predict([img_batch, clip_batch], verbose=0)
                
                pred_items = predictions[0][0] # Item probabilities
                pred_count = predictions[1][0][0] # Count regression
                pred_quantity = predictions[2][0][0] # Qty regression
                
                # 4. Process Results
                st.metric("Estimated Item Count", f"{round(pred_count)}")
                st.metric("Total Quantity", f"{round(pred_quantity)}")
                
                st.subheader("Detected Items")
                
                # Get top K predictions
                top_k = 10
                idx_sorted = np.argsort(-pred_items)[:top_k]
                
                results_data = []
                for idx in idx_sorted:
                    prob = float(pred_items[idx])
                    if prob > 0.1: # Threshold to show
                        asin = processor.idx_to_asin.get(int(idx), 'Unknown')
                        name = processor.asin_to_name.get(asin, 'Unknown Product')
                        results_data.append({
                            "ASIN": asin,
                            "Product": name,
                            "Confidence": f"{prob:.2%}"
                        })
                
                if results_data:
                    st.table(results_data)
                else:
                    st.write("No items detected with high confidence.")