import streamlit as st
import os
import sys
import pickle
import time
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
import gdown  # <--- CRITICAL for large Drive files

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Your Links
MODEL_FILE_ID = "1zOifXFl_joyEsmRkVpVSNYYwvFzF1Uoo"  # Extracted ID from your link
VOCAB_FILE_ID = "1CbkG7fRFHKhd4VbZ82zd1B-MpEEzIF7n"  # Extracted ID from your link

MODEL_PATH = "bin_classifier_10k_subset_best.h5"
VOCAB_PATH = "asin_vocabulary_10k_clean.pkl"

st.set_page_config(page_title="Amazon Bin Classifier", layout="wide")

# ==========================================
# 2. ROBUST DOWNLOADER (Auto-Fixes Fake Files)
# ==========================================

def download_asset(file_id, output_path, expected_size_mb=0):
    """
    Downloads from Drive using ID. 
    Auto-deletes file if it's too small (fake HTML page).
    """
    # 1. Check existing file
    if os.path.exists(output_path):
        # If we expect a large model (>10MB) and found a tiny file (<1MB), it's fake.
        if expected_size_mb > 10 and os.path.getsize(output_path) < 1024 * 1024:
            try:
                os.remove(output_path)
                st.warning(f"🗑️ Deleted corrupt/incomplete file: {output_path}")
            except PermissionError:
                st.error(f"❌ File '{output_path}' is corrupt but LOCKED. Please stop the app (Ctrl+C) and delete it manually.")
                st.stop()
        else:
            return True # File looks okay

    # 2. Download using gdown (Handles Virus Warning automatically)
    url = f'https://drive.google.com/uc?id={file_id}'
    with st.spinner(f"Downloading {output_path} (approx {expected_size_mb} MB)..."):
        try:
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            st.error(f"Download Error: {e}")
            return False

    # 3. Validate Download
    if not os.path.exists(output_path):
        st.error(f"❌ Failed to download {output_path}")
        return False
        
    # Final Size Check
    if expected_size_mb > 10 and os.path.getsize(output_path) < 1024 * 1024:
        st.error(f"❌ Downloaded file is too small! Google Drive likely blocked the download.")
        st.info("Check if the file permission is set to 'Anyone with the link'.")
        os.remove(output_path) # Cleanup
        return False

    return True

# ==========================================
# 3. RESOURCE LOADING
# ==========================================

@st.cache_resource
def load_resources():
    # 1. Download Model (Expect ~180MB)
    if not download_asset(MODEL_FILE_ID, MODEL_PATH, expected_size_mb=180):
        st.stop()
        
    # 2. Download Vocab (Expect ~1MB)
    if not download_asset(VOCAB_FILE_ID, VOCAB_PATH, expected_size_mb=1):
        st.stop()

    # 3. Load Vocab
    try:
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        processor = AmazonBinDataProcessor(vocab)
    except Exception as e:
        st.error(f"Error loading vocabulary: {e}")
        st.stop()

    # 4. Load Model
    # ... inside load_resources() ...

    # 4. Load Model
    try:
        # Option A: Try loading the full model (architecture + weights)
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Loaded full model successfully.")
    except Exception as e:
        print(f"⚠️ Full model load failed ({e}), switching to weights-only mode...")
        
        # Option B: Fallback to building architecture + loading weights
        # 1. Get the exact number of classes from your NEW vocabulary
        vocab_len = len(processor.asin_to_idx)
        
        # 2. Build the model structure with the correct number of outputs
        model = build_hybrid_model(vocab_len)
        
        # 3. Load the weights
        try:
            model.load_weights(MODEL_PATH)
            print(f"✅ Loaded weights for {vocab_len} classes.")
        except Exception as w_e:
            st.error(f"❌ Shape Mismatch! Your 'vocab.pkl' has {vocab_len} classes, but the model weights expect a different number.")
            st.error(f"Details: {w_e}")
            st.stop()

    # 5. Load CLIP
    clip_extractor = CLIPFeatureExtractor()

    return clip_extractor, processor, model

# ==========================================
# 4. CLASSES
# ==========================================

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
    def __init__(self, vocab):
        self.img_size = (224, 224)
        self.asin_to_idx = vocab.get('asin_to_idx', {})
        self.idx_to_asin = vocab.get('idx_to_asin', {})
        self.asin_to_name = vocab.get('asin_to_name', {})

    def preprocess(self, image_pil):
        # 1. Convert PIL to NumPy
        img = np.array(image_pil)
        
        # 2. Ensure RGB (PIL is RGB, but just in case)
        if img.ndim == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4: # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        # 3. APPLY THE SAME FILTERS AS TRAINING (Critical!)
        try:
            # Bilateral Filter (Noise Removal)
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # CLAHE (Contrast Enhancement)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            enhanced = cv2.merge([l2, a, b])
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        except Exception:
            pass # Fallback if filters fail

        # 4. Resize and Normalize
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

    item_branch = layers.Dense(2048, activation='relu')(fused)
    item_branch = layers.Dropout(0.3)(item_branch)
    item_output = layers.Dense(num_classes, activation='sigmoid', name='item_presence')(item_branch)

    count_branch = layers.Dense(128, activation='relu')(fused)
    count_branch = layers.Dropout(0.3)(count_branch)
    count_output = layers.Dense(1, activation='relu', name='item_count')(count_branch)

    quantity_branch = layers.Dense(128, activation='relu')(fused)
    quantity_branch = layers.Dropout(0.3)(quantity_branch)
    quantity_output = layers.Dense(1, activation='relu', name='total_quantity')(quantity_branch)
    return Model(inputs=[cnn_input, clip_input], outputs=[item_output, count_output, quantity_output])

# ==========================================
# 5. MAIN UI
# ==========================================

st.title("📦 Amazon Bin Classifier")

# Load Resources
clip_extractor, processor, model = load_resources()

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
            
            try:
                clip_features = clip_extractor.extract(image_pil)
                clip_batch = np.expand_dims(clip_features, axis=0)
            except Exception as e:
                clip_batch = np.zeros((1, 512))
            
            predictions = model.predict([img_batch, clip_batch], verbose=0)
            
            pred_items = predictions[0][0]
            # DEBUG: Print the top 3 raw scores to the screen
            st.write("--- Debugging Raw Scores ---")
            top_3_indices = np.argsort(-pred_items)[:3]
            for idx in top_3_indices:
                score = float(pred_items[idx])
                name = processor.asin_to_name.get(processor.idx_to_asin.get(int(idx)), "Unknown")
                st.write(f"Top guess: **{name}** with score: **{score:.4f}**")
            st.write("----------------------------")
            pred_count = predictions[1][0][0]
            pred_quantity = predictions[2][0][0]
            
            st.metric("Estimated Count", f"{round(pred_count)}")
            st.metric("Total Quantity", f"{round(pred_quantity)}")
            
            st.subheader("Items")
            idx_sorted = np.argsort(-pred_items)[:10]
            results = []
            for idx in idx_sorted:
                if float(pred_items[idx]) > 0.05:
                    asin = processor.idx_to_asin.get(int(idx), 'Unknown')
                    name = processor.asin_to_name.get(asin, 'Unknown Product')
                    results.append(f"**{name}** ({float(pred_items[idx]):.0%})")
            
            if results:
                for r in results: st.write(r)
            else:
                st.write("No items confident.")
