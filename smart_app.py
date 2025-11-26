import streamlit as st
import os
import requests
import sys
import pickle
import shutil
import time
import re

# ==========================================
# CONFIGURATION
# ==========================================
# Your Links
MODEL_URL = "https://drive.google.com/uc?id=1g5deJ3i0h1843BXCO-Kn-2xRvaNzEsCb"  
VOCAB_URL = "https://drive.google.com/uc?id=1ET4ap1LK-yjr891U-WVJZT0oT1gwascD"

MODEL_PATH = "demo_bin_classifier_best.h5"
VOCAB_PATH = "asin_vocabulary.pkl"

st.set_page_config(page_title="Amazon Bin Classifier", layout="wide")

# ==========================================
# DOWNLOAD UTILITIES (Smarter Validation)
# ==========================================

def transform_drive_url(url):
    """Converts a standard Google Drive 'view' link to a direct download link."""
    if "drive.google.com" not in url:
        return url
    patterns = [r'/file/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
    for p in patterns:
        match = re.search(p, url)
        if match:
            return f'https://drive.google.com/uc?export=download&id={match.group(1)}'
    return url

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_file_from_google_drive(url, destination):
    session = requests.Session()
    response = session.get(url, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)    

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: f.write(chunk)

def robust_download(url, filename):
    # 1. Download if missing
    if not os.path.exists(filename):
        if "YOUR_DIRECT_LINK" in url: return False
        
        clean_url = transform_drive_url(url)
        with st.spinner(f"Downloading {filename}..."):
            try:
                if "drive.google.com" in clean_url:
                    download_file_from_google_drive(clean_url, filename)
                else:
                    response = requests.get(clean_url, stream=True)
                    save_response_content(response, filename)
            except Exception as e:
                st.error(f"Network error downloading {filename}: {e}")
                return False

    # 2. VALIDATION (The Fix)
    if not os.path.exists(filename): return False
    
    # Check 1: Empty file?
    if os.path.getsize(filename) < 100:
        os.remove(filename)
        st.error(f"Download failed: {filename} is empty.")
        return False

    # Check 2: File specific validation
    if filename.endswith(".pkl"):
        try:
            with open(filename, 'rb') as f:
                pickle.load(f)
        except Exception:
            os.remove(filename)
            st.error(f"Error: {filename} is corrupted (likely an HTML error page). Checking permissions...")
            return False
            
    elif filename.endswith(".h5"):
        # We delay h5 validation to the main loader to avoid double import, 
        # but we check simple size (h5 model should be big, >1MB)
        if os.path.getsize(filename) < 1024 * 1024: # 1MB
             # Try reading first few bytes to see if it's HTML
             with open(filename, 'rb') as f:
                 header = f.read(100)
                 if b'<!DOCTYPE html>' in header or b'<html' in header:
                     os.remove(filename)
                     st.error(f"Error: {filename} is an HTML page, not a model. Check Link Permissions.")
                     return False

    return True

# ==========================================
# HEAVY LOADING
# ==========================================

@st.cache_resource
def load_heavy_libraries():
    with st.spinner("Loading AI Engines (TensorFlow, PyTorch)..."):
        import numpy as np
        import cv2
        from PIL import Image
        import h5py
        import torch
        import clip
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, Model
        from tensorflow.keras.applications import EfficientNetV2B0
        return np, cv2, Image, h5py, torch, clip, tf, keras, layers, Model, EfficientNetV2B0

@st.cache_resource
def setup_model(vocab_url, model_url):
    # 1. Download with smart validation
    if not robust_download(vocab_url, VOCAB_PATH): return None, None, None, "Vocab download failed."
    if not robust_download(model_url, MODEL_PATH): return None, None, None, "Model download failed."

    # 2. Load Libs
    try:
        np, cv2, Image, h5py, torch, clip, tf, keras, layers, Model, EfficientNetV2B0 = load_heavy_libraries()
    except Exception as e:
        return None, None, None, f"Library Import Error: {str(e)}"

    # 3. Final Integrity Check
    if not h5py.is_hdf5(MODEL_PATH):
        os.remove(MODEL_PATH)
        return None, None, None, "Corrupt .h5 file (Invalid HDF5). Deleted. Please reload."

    # --- Classes ---
    class CLIPFeatureExtractor:
        def __init__(self):
            self.device = "cpu"
            self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
            self.model.eval()
        def extract(self, image_pil):
            with torch.no_grad():
                image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
                features = self.model.encode_image(image_input)
                features = features.cpu().numpy().flatten()
                norm = np.linalg.norm(features) + 1e-10
                features = features / norm
            return features

    class AmazonBinDataProcessor:
        def __init__(self, vocab_path):
            self.img_size = (224, 224)
            try:
                with open(vocab_path, 'rb') as f:
                    vocab = pickle.load(f)
                self.asin_to_idx = vocab.get('asin_to_idx', {})
                self.idx_to_asin = vocab.get('idx_to_asin', {})
                self.asin_to_name = vocab.get('asin_to_name', {})
            except:
                self.asin_to_idx = {}
                self.idx_to_asin = {}
                self.asin_to_name = {}
        def preprocess(self, image_pil):
            img = np.array(image_pil)
            img = cv2.resize(img, tuple(self.img_size))
            img = img.astype(np.float32) / 255.0
            return img

    def build_hybrid_model(num_classes):
        cnn_input = layers.Input(shape=(224, 224, 3), name='cnn_input')
        backbone = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=cnn_input)
        cnn_features = backbone.output
        cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
        cnn_features = layers.Dropout(0.3)(cnn_features)
        cnn_features = layers.Dense(512, activation='relu', name='cnn_dense')(cnn_features)
        cnn_features = layers.BatchNormalization()(cnn_features)

        clip_input = layers.Input(shape=(512,), name='clip_input')
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
        return Model(inputs=[cnn_input, clip_input], outputs=[item_output, count_output, quantity_output])

    # --- Loading Logic ---
    clip_extractor = CLIPFeatureExtractor()
    processor = AmazonBinDataProcessor(VOCAB_PATH)
    
    try:
        model = keras.models.load_model(MODEL_PATH)
    except:
        # Fallback Logic: Try different vocab sizes to match weights
        vocab_len = len(processor.asin_to_idx) if processor.asin_to_idx else 0
        
        # We try the calculated vocab size first, then 200 (the training script limit)
        # This fixes "Weight Mismatch: shape (256, 200) vs (256, 100)" errors
        candidate_sizes = [min(200, max(1, vocab_len)), 200]
        candidate_sizes = sorted(list(set(candidate_sizes))) # dedupe
        
        success = False
        last_err = None
        
        for n in candidate_sizes:
            try:
                model = build_hybrid_model(n)
                model.load_weights(MODEL_PATH)
                success = True
                break
            except Exception as e:
                last_err = e
        
        if not success:
            return None, None, None, f"Weight Mismatch: {str(last_err)}"

    return clip_extractor, processor, model, None

# ==========================================
# MAIN UI
# ==========================================

st.title("📦 Amazon Bin Classifier")

if "YOUR_DIRECT_LINK" in MODEL_URL:
    st.error("⚠️ Please add links in app.py")
    st.stop()

clip_extractor, processor, model, error_msg = setup_model(VOCAB_URL, MODEL_URL)

if error_msg:
    st.error(f"Initialization Failed: {error_msg}")
elif model:
    try:
        import numpy as np
        from PIL import Image
    except:
        st.stop()

    uploaded_file = st.file_uploader("Choose a bin image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.subheader("Analysis")
            with st.spinner("Processing..."):
                img_processed = processor.preprocess(image_pil)
                img_batch = np.expand_dims(img_processed, axis=0)
                clip_features = clip_extractor.extract(image_pil)
                clip_batch = np.expand_dims(clip_features, axis=0)
                
                predictions = model.predict([img_batch, clip_batch], verbose=0)
                
                pred_items = predictions[0][0]
                pred_count = predictions[1][0][0]
                pred_quantity = predictions[2][0][0]
                
                st.metric("Estimated Count", f"{round(pred_count)}")
                st.metric("Total Quantity", f"{round(pred_quantity)}")
                
                st.subheader("Items")
                idx_sorted = np.argsort(-pred_items)[:10]
                results = []
                for idx in idx_sorted:
                    if float(pred_items[idx]) > 0.1:
                        asin = processor.idx_to_asin.get(int(idx), 'Unknown')
                        name = processor.asin_to_name.get(asin, 'Unknown Product')
                        results.append(f"**{name}** ({float(pred_items[idx]):.0%})")
                
                if results:
                    for r in results: st.write(r)
                else:
                    st.write("No items confident.")
