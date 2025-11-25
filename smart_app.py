import streamlit as st
import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pickle
from datetime import datetime
import tempfile

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras

# CLIP Integration
import torch
import clip

# Import your SmartClip modules
# Assuming smartclip.py is in the same directory or in Python path
try:
    from smartclip import (
        CLIPFeatureExtractor,
        AmazonBinDataProcessor,
        HybridCLIPBinClassifier,
        CLIPEnhancedBinValidator,
        load_model,
        predict_single
    )
except ImportError:
    st.error("⚠️ Could not import smartclip module. Ensure smartclip.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SmartClip - Bin Validation System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'validator' not in st.session_state:
    st.session_state.validator = None
if 'model' not in st.session_state:
    st.session_state.model = None


@st.cache_resource
def load_system(model_path, vocab_path, use_clip=True):
    """Load the model and processor (cached)"""
    try:
        # Create a minimal processor without needing image/metadata directories
        # We'll create a temporary directory structure
        temp_dir = Path(tempfile.gettempdir()) / "smartclip_temp"
        temp_dir.mkdir(exist_ok=True)
        
        processor = AmazonBinDataProcessor(
            images_dir=str(temp_dir),
            metadata_dir=str(temp_dir),
            img_size=(224, 224),
            use_clip=use_clip
        )
        processor.load_vocabulary(vocab_path)
        
        # Load model with custom objects to handle serialization issues
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError(),
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
        }
        
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Recompile the model with proper metrics
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss={
                'item_presence': 'binary_crossentropy',
                'item_count': 'mse',
                'total_quantity': 'mse'
            },
            loss_weights={
                'item_presence': 5.0,
                'item_count': 1.0,
                'total_quantity': 1.0
            },
            metrics={
                'item_presence': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'item_count': ['mae'],
                'total_quantity': ['mae']
            }
        )
        
        # Create validator
        validator = CLIPEnhancedBinValidator(
            model=model,
            processor=processor,
            threshold=0.5,
            use_clip_similarity=use_clip
        )
        
        return model, processor, validator, None
    except Exception as e:
        return None, None, None, str(e)


def main():
    # Header
    st.markdown('<p class="main-header">📦 SmartClip - AI-Powered Bin Validation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Validate warehouse bin contents using Computer Vision + CLIP</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Model Settings")
        model_path = st.text_input(
            "Model Path (.h5)",
            value="models/demo_bin_classifier_best.h5",
            help="Path to your trained model file"
        )
        
        vocab_path = st.text_input(
            "Vocabulary Path (.pkl)",
            value="asin_vocabulary.pkl",
            help="Path to ASIN vocabulary pickle file"
        )
        
        use_clip = st.checkbox("Use CLIP Features", value=True, help="Enable CLIP-enhanced predictions")
        
        st.divider()
        
        load_button = st.button("🚀 Load Model", type="primary", use_container_width=True)
        
        if load_button:
            with st.spinner("Loading model and initializing system..."):
                model, processor, validator, error = load_system(
                    model_path, vocab_path, use_clip
                )
                
                if error:
                    st.error(f"❌ Error loading system: {error}")
                    st.session_state.model_loaded = False
                else:
                    st.session_state.model = model
                    st.session_state.processor = processor
                    st.session_state.validator = validator
                    st.session_state.model_loaded = True
                    st.success("✅ System loaded successfully!")
                    st.rerun()
        
        # Status indicator
        st.divider()
        if st.session_state.model_loaded:
            st.success("✅ System Ready")
            if st.session_state.processor:
                st.metric("Total ASINs", len(st.session_state.processor.asin_to_idx))
        else:
            st.warning("⚠️ Please load the model")
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Please configure and load the model from the sidebar to begin.")
        
        # Show sample configuration
        with st.expander("📖 Setup Instructions"):
            st.markdown("""
            ### How to Get Started:
            
            1. **Model File**: Ensure your trained model (.h5 file) is available
            2. **Vocabulary File**: Make sure asin_vocabulary.pkl exists in the same directory or provide full path
            3. **Load System**: Click the "Load Model" button
            
            ### Required Files:
            - `demo_bin_classifier_best.h5` (or your model name)
            - `asin_vocabulary.pkl`
            
            ### Note:
            The application now works independently - you only need to upload images for validation!
            """)
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Order Validation",
        "🔍 Single Prediction",
        "📊 Batch Processing",
        "ℹ️ System Info"
    ])
    
    # TAB 1: Order Validation
    with tab1:
        st.header("Order Validation")
        st.markdown("Upload a bin image and enter the expected order items to validate.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Upload Bin Image")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                key="validation_upload"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("📝 Expected Order Items")
            
            # Option to input order items
            input_method = st.radio(
                "Input Method:",
                ["Manual Entry", "JSON Upload"],
                horizontal=True
            )
            
            order_items = {}
            
            if input_method == "Manual Entry":
                st.markdown("Enter ASIN and quantity for each item:")
                
                num_items = st.number_input("Number of items", min_value=1, max_value=20, value=3)
                
                for i in range(num_items):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        asin = st.text_input(f"ASIN {i+1}", key=f"asin_{i}")
                    with col_b:
                        qty = st.number_input(f"Qty {i+1}", min_value=1, value=1, key=f"qty_{i}")
                    
                    if asin:
                        order_items[asin] = qty
            
            else:  # JSON Upload
                json_file = st.file_uploader(
                    "Upload JSON order file",
                    type=['json'],
                    key="json_upload"
                )
                if json_file:
                    try:
                        order_items = json.load(json_file)
                        st.json(order_items)
                    except Exception as e:
                        st.error(f"Error parsing JSON: {e}")
            
            st.divider()
            
            # Confidence threshold
            threshold = st.slider(
                "Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score to consider an item as present"
            )
        
        # Validate button
        if st.button("🔍 Validate Order", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload an image!")
            elif not order_items:
                st.error("Please enter at least one order item!")
            else:
                with st.spinner("Analyzing bin contents..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Update validator threshold
                        st.session_state.validator.threshold = threshold
                        
                        # Perform validation
                        result = st.session_state.validator.validate_order(tmp_path, order_items)
                        
                        # Display results
                        st.divider()
                        
                        if result.get('valid'):
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("✅ **VALIDATION PASSED** - All items present!")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box">', unsafe_allow_html=True)
                            st.error("❌ **VALIDATION FAILED** - Some items missing!")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Items Present",
                                len(result.get('items_present', [])),
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "Items Missing",
                                len(result.get('items_missing', [])),
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Predicted Count",
                                result.get('predicted_item_count', 0),
                                delta=f"{result.get('predicted_item_count', 0) - result.get('expected_item_count', 0)}"
                            )
                        
                        with col4:
                            st.metric(
                                "Predicted Quantity",
                                result.get('predicted_total_quantity', 0),
                                delta=f"{result.get('predicted_total_quantity', 0) - result.get('expected_total_quantity', 0)}"
                            )
                        
                        # Detailed results
                        st.subheader("📋 Detailed Results")
                        
                        results_df = pd.DataFrame(result.get('detailed_results', []))
                        if not results_df.empty:
                            results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.3f}")
                            results_df['status'] = results_df['detected'].apply(lambda x: "✅ Present" if x else "❌ Missing")
                            
                            st.dataframe(
                                results_df[['asin', 'product_name', 'ordered_quantity', 'status', 'confidence']],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        # Missing items alert
                        if result.get('items_missing'):
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.warning("**Missing Items:**")
                            for asin in result['items_missing']:
                                product_name = st.session_state.processor.asin_to_name.get(asin, 'Unknown')
                                st.write(f"- **{asin}**: {product_name}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download report
                        st.divider()
                        report = {
                            'timestamp': datetime.now().isoformat(),
                            'validation_result': result,
                            'order_items': order_items
                        }
                        
                        st.download_button(
                            label="📥 Download Validation Report (JSON)",
                            data=json.dumps(report, indent=2),
                            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    except Exception as e:
                        st.error(f"Error during validation: {e}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    # TAB 2: Single Prediction
    with tab2:
        st.header("Single Image Prediction")
        st.markdown("Upload an image to see what items the model detects.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Upload Image")
            pred_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                key="prediction_upload"
            )
            
            if pred_file:
                image = Image.open(pred_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            top_k = st.slider("Number of top predictions", min_value=5, max_value=50, value=10)
        
        with col2:
            if st.button("🔮 Predict Items", type="primary", use_container_width=True):
                if not pred_file:
                    st.error("Please upload an image!")
                else:
                    with st.spinner("Making predictions..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(pred_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            results = predict_single(
                                st.session_state.model,
                                st.session_state.processor,
                                tmp_path,
                                top_k=top_k
                            )
                            
                            st.subheader(f"🎯 Top {top_k} Predictions")
                            
                            # Create dataframe
                            pred_df = pd.DataFrame(results, columns=['ASIN', 'Confidence', 'Product Name'])
                            pred_df['Confidence'] = pred_df['Confidence'].apply(lambda x: f"{x:.4f}")
                            pred_df['Rank'] = range(1, len(pred_df) + 1)
                            
                            # Reorder columns
                            pred_df = pred_df[['Rank', 'ASIN', 'Product Name', 'Confidence']]
                            
                            st.dataframe(pred_df, use_container_width=True, hide_index=True)
                            
                            # Visualization
                            st.subheader("📊 Confidence Distribution")
                            
                            chart_data = pd.DataFrame({
                                'Item': [f"{r[0][:10]}..." for r in results[:10]],
                                'Confidence': [r[1] for r in results[:10]]
                            })
                            
                            st.bar_chart(chart_data.set_index('Item'))
                        
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                        
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
    
    # TAB 3: Batch Processing
    with tab3:
        st.header("Batch Processing")
        st.markdown("Upload multiple images for batch validation.")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_list = []
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        predictions = predict_single(
                            st.session_state.model,
                            st.session_state.processor,
                            tmp_path,
                            top_k=5
                        )
                        
                        results_list.append({
                            'filename': file.name,
                            'top_prediction': predictions[0][0] if predictions else 'N/A',
                            'confidence': predictions[0][1] if predictions else 0.0,
                            'product_name': predictions[0][2] if predictions else 'N/A'
                        })
                    
                    except Exception as e:
                        results_list.append({
                            'filename': file.name,
                            'top_prediction': 'ERROR',
                            'confidence': 0.0,
                            'product_name': str(e)
                        })
                    
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing complete!")
                
                # Display results
                st.subheader("📊 Batch Results")
                results_df = pd.DataFrame(results_list)
                st.dataframe(results_df, use_container_width=True)
                
                # Download CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # TAB 4: System Info
    with tab4:
        st.header("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Statistics")
            if st.session_state.processor:
                st.metric("Total ASINs", len(st.session_state.processor.asin_to_idx))
                st.metric("Products with Names", len(st.session_state.processor.asin_to_name))
                st.metric("Image Size", f"{st.session_state.processor.img_size[0]}x{st.session_state.processor.img_size[1]}")
                st.metric("CLIP Enabled", "Yes" if st.session_state.processor.use_clip else "No")
        
        with col2:
            st.subheader("🔧 System Configuration")
            st.text(f"TensorFlow: {tf.__version__}")
            st.text(f"PyTorch: {torch.__version__}")
            st.text(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.text(f"GPU: {torch.cuda.get_device_name(0)}")
        
        st.divider()
        
        st.subheader("📖 Sample ASINs")
        if st.session_state.processor and st.session_state.processor.asin_to_name:
            sample_asins = list(st.session_state.processor.asin_to_name.items())[:20]
            sample_df = pd.DataFrame(sample_asins, columns=['ASIN', 'Product Name'])
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("ℹ️ About SmartClip")
        st.markdown("""
        **SmartClip** is an AI-powered warehouse bin validation system that combines:
        
        - **Computer Vision**: EfficientNetV2B0 CNN for visual feature extraction
        - **CLIP Integration**: OpenAI's CLIP for enhanced multi-modal understanding
        - **Multi-task Learning**: Simultaneous prediction of item presence, count, and quantity
        
        This system helps warehouse operations ensure order accuracy and reduce picking errors.
        """)


if __name__ == "__main__":
    main()