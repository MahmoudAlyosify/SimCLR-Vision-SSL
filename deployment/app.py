"""
================================================================================
  SimCLR ResNet-50 Visual Search Engine GUI
  -----------------------------------------
  A premium, interactive web-based graphical user interface (GUI) built with
  Streamlit, powered by ONNX Runtime and FAISS for real-time visual similarity
  retrieval on CIFAR-10.

  Key Features:
    1. Premium Dark Mode UI with tailored CSS glassmorphism styling
    2. Upload any local image or select a random test image from CIFAR-10
    3. Lightning-fast feature extraction (2048-d) via optimized ONNX Runtime
    4. Sub-millisecond exact Cosine Similarity search using FAISS vector database
    5. Clean visual results grid with similarity scores and class matches
    6. Dedicated interactive "Ablation Study Dashboard" showing the Exp 38-42 findings

  Usage:
    streamlit run app.py
================================================================================
"""

import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image

import streamlit as st

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# ==========================================================================
# Caching Assets for Instant Performance
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(BASE_DIR, "deployment")
ONNX_PATH = os.path.join(DEPLOY_DIR, "simclr_encoder_exp41.onnx")
INDEX_PATH = os.path.join(DEPLOY_DIR, "cifar10_index.faiss")
META_PATH = os.path.join(DEPLOY_DIR, "metadata.json")

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


@st.cache_resource
def load_onnx_model():
    """Load ONNX inference session and cache it."""
    import onnxruntime as ort
    # CPU is fast enough for single image inference, ensures compatibility
    providers = ['CPUExecutionProvider']
    return ort.InferenceSession(ONNX_PATH, providers=providers)


@st.cache_resource
def load_faiss_index():
    """Load FAISS index and cache it."""
    import faiss
    return faiss.read_index(INDEX_PATH)


@st.cache_data
def load_metadata():
    """Load metadata dictionary and cache it."""
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -- Check that deployment files exist ---------------------------------------
missing_files = []
for p, name in [(ONNX_PATH, "ONNX Model"), (INDEX_PATH, "FAISS Index"), (META_PATH, "Metadata JSON")]:
    if not os.path.exists(p):
        missing_files.append(name)

if missing_files:
    st.error(f"❌ Missing deployment assets: {', '.join(missing_files)}")
    st.warning("👉 Please run `export_onnx.py` then `build_faiss.py` to generate these files before running the GUI!")
    st.stop()

# Load cached assets
ort_sess = load_onnx_model()
faiss_index = load_faiss_index()
metadata = load_metadata()

# ==========================================================================
# Preprocessing Logic
# ==========================================================================
def preprocess_image(pil_img):
    """
    Resizes image to 32x32 and applies CIFAR-10 mean/std normalization.
    Returns: numpy array of shape (1, 3, 32, 32)
    """
    # 1. Resize to 32x32
    img_resized = pil_img.resize((32, 32), Image.Resampling.BILINEAR)
    
    # 2. Convert to numpy array and scale to [0, 1]
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Handle grayscale images if uploaded
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3] # Drop alpha channel
        
    # 3. Normalize: (x - mean) / std
    img_normalized = (img_np - CIFAR_MEAN) / CIFAR_STD
    
    # 4. Transpose to channel-first (HWC -> CHW) and add batch dimension (1, C, H, W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch


def perform_search(features_batch, top_k=5):
    """
    Normalizes features, queries FAISS, and returns matched metadata.
    """
    # 1. L2-Normalize vector
    norm = np.linalg.norm(features_batch, axis=1, keepdims=True)
    features_normalized = features_batch / np.maximum(norm, 1e-12)
    
    # 2. Search FAISS Index
    similarities, indices = faiss_index.search(features_normalized, top_k)
    
    results = []
    for rank in range(top_k):
        vector_id = int(indices[0][rank])
        score = float(similarities[0][rank])
        
        # Load from metadata catalog
        match_info = metadata["images"][vector_id]
        
        results.append({
            "rank": rank + 1,
            "id": vector_id,
            "similarity": score,
            "image_path": os.path.join(DEPLOY_DIR, match_info["image_path"]),
            "class_name": match_info["class_name"],
            "label_id": match_info["label_id"]
        })
        
    return results


# ==========================================================================
# Streamlit UI Configuration
# ==========================================================================
st.set_page_config(
    page_title="SimCLR ResNet-50 Visual Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Styling
st.markdown("""
<style>
    /* Dark theme customizations */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    
    /* Title and Header designs */
    h1 {
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    
    /* Custom Card/Glassmorphism block */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# Sidebar Configuration
# ==========================================================================
st.sidebar.markdown("### ⚙️ Engine Settings")
top_k_slider = st.sidebar.slider("Number of results (K)", min_value=3, max_value=20, value=6, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Trained Backbone Model")
st.sidebar.markdown("""
* **Architecture**: ResNet-50 (CIFAR STEM adjusted)
* **Training Type**: Self-Supervised (SimCLR)
* **Best Experiment**: **Exp 41**
* **Augmentations**: Crop + Flip + Blur + Color Jitter
* **Midterm Baseline**: 64.49%
* **Final Linear Probe Accuracy**: **84.30%** (+19.81% Gain!)
* **Embedding Dimension**: 2048
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Backend Pipeline")
st.sidebar.markdown("""
* **Inference Engine**: ONNX Runtime CPU
* **Vector Database**: FAISS (IndexFlatIP)
* **Similarity Metric**: Exact Cosine Similarity
* **Database Size**: 10,000 Images
""")

st.sidebar.caption("Group 20, CISC 867, Queen's University, Spring 2026")

# ==========================================================================
# Main Title
# ==========================================================================
st.title("🔍 Real-time SimCLR Image Retrieval Engine")
st.markdown("##### Self-Supervised Representation Learning with ResNet-50 & FAISS Indexing")

# Define Tabs
tab1, tab2 = st.tabs(["🚀 Real-Time Search", "📊 Ablation Study Dashboard"])

# ==========================================================================
# TAB 1: Visual Search Engine
# ==========================================================================
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")
    
    # Initialize session state for random query image selection
    if "random_img_id" not in st.session_state:
        st.session_state.random_img_id = None
        
    query_image = None
    query_info = None

    with col_left:
        st.markdown("### 📥 Select Query Image")
        
        # Method 1: Upload a file
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        
        # Method 2: Pick a random image from the database
        st.markdown("<p style='text-align: center; color: #94A3B8; margin: 10px 0;'>— OR —</p>", unsafe_allow_html=True)
        if st.button("🎲 Pick a Random Test Image from Database", use_container_width=True):
            st.session_state.random_img_id = random.randint(0, len(metadata["images"]) - 1)
            
        # Display selected/uploaded image
        if uploaded_file is not None:
            st.session_state.random_img_id = None # Clear random image
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Uploaded Query Image", use_container_width=True)
            
        elif st.session_state.random_img_id is not None:
            idx = st.session_state.random_img_id
            img_info = metadata["images"][idx]
            local_path = os.path.join(DEPLOY_DIR, img_info["image_path"])
            
            if os.path.exists(local_path):
                query_image = Image.open(local_path).convert("RGB")
                query_info = img_info
                st.image(
                    query_image, 
                    caption=f"Selected Reference Image #{idx} (Class: {img_info['class_name']})", 
                    use_container_width=True
                )
            else:
                st.error("Reference image file not found.")

    with col_right:
        st.markdown("### 🎯 Visual Similarity Search Results")
        
        if query_image is not None:
            t_start = time.time()
            
            # 1. Preprocess
            batch_img = preprocess_image(query_image)
            
            # 2. ONNX Inference
            onnx_outputs = ort_sess.run(None, {"images": batch_img})
            feature_vector = onnx_outputs[0] # (1, 2048)
            
            # 3. FAISS Query
            t_search_start = time.time()
            results = perform_search(feature_vector, top_k=top_k_slider)
            
            t_total_ms = (time.time() - t_start) * 1000
            t_search_ms = (time.time() - t_search_start) * 1000
            
            # Show Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f'<div class="glass-card"><div class="metric-value">{t_search_ms:.2f} ms</div><div class="metric-label">FAISS Query Time</div></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="glass-card"><div class="metric-value">{t_total_ms:.1f} ms</div><div class="metric-label">Total Pipeline Latency</div></div>', unsafe_allow_html=True)
            with m_col3:
                # Calculate class precision
                if query_info is not None:
                    matches = sum(1 for r in results if r["class_name"] == query_info["class_name"])
                    precision = (matches / len(results)) * 100
                    st.markdown(f'<div class="glass-card"><div class="metric-value">{precision:.1f}%</div><div class="metric-label">Top-{len(results)} Class Precision</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="glass-card"><div class="metric-value">N/A</div><div class="metric-label">Upload class unknown</div></div>', unsafe_allow_html=True)
            
            # Display Results Grid
            st.markdown("#### Retreived Nearest Neighbors")
            
            # Create dynamic grid columns
            grid_cols = st.columns(3)
            
            for idx, res in enumerate(results):
                col = grid_cols[idx % 3]
                
                with col:
                    # Determine styling color based on class match if query info exists
                    is_match = False
                    border_color = "rgba(255, 255, 255, 0.05)"
                    bg_color = "rgba(30, 41, 59, 0.7)"
                    badge_html = f'<span style="background-color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; color: white;">{res["class_name"]}</span>'
                    
                    if query_info is not None:
                        is_match = (res["class_name"] == query_info["class_name"])
                        if is_match:
                            border_color = "rgba(16, 185, 129, 0.3)"
                            bg_color = "rgba(6, 78, 59, 0.3)"
                            badge_html = f'<span style="background-color: #10B981; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; color: white;">{res["class_name"]} (MATCH)</span>'
                        else:
                            border_color = "rgba(239, 68, 68, 0.2)"
                            bg_color = "rgba(127, 29, 29, 0.1)"
                            badge_html = f'<span style="background-color: #EF4444; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; color: white;">{res["class_name"]}</span>'
                    
                    # Custom card container
                    card_content = f"""
                    <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center;">
                        <div style="color: #38BDF8; font-size: 1.1rem; font-weight: 700; margin-bottom: 8px;">Rank #{res["rank"]}</div>
                        <div style="margin-bottom: 10px;">{badge_html}</div>
                        <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 4px;">Cosine Similarity:</div>
                        <div style="color: #E2E8F0; font-size: 1.4rem; font-weight: 700; margin-bottom: 10px;">{res["similarity"] * 100:.2f}%</div>
                        <div style="color: #64748B; font-size: 0.8rem;">Vector ID: {res["id"]}</div>
                    </div>
                    """
                    
                    # Render matched image and custom card
                    if os.path.exists(res["image_path"]):
                        matched_img = Image.open(res["image_path"])
                        col.image(matched_img, use_container_width=True)
                    else:
                        col.error("Image file missing.")
                        
                    col.markdown(card_content, unsafe_allow_html=True)
                    
        else:
            st.info("💡 Please upload a custom image or click the button to select a random test image to query the visual search engine!")

# ==========================================================================
# TAB 2: Ablation Study Dashboard
# ==========================================================================
with tab2:
    st.markdown("### 📊 Experiment Ablation Study & Color Jitter Findings")
    st.markdown("##### Evaluating Natalie's Midterm Pipelines against Mahmoud's Final Jittered Re-runs")
    
    st.markdown("""
    This section presents the results of the complete **Color Jitter Shortcut-Learning Ablation Study** (Experiments 38-42).
    By comparing the self-supervised representations learned with and without photometric distortion (Color Jitter), we demonstrate how SimCLR ResNet-50 encoders learn rich semantic representations instead of exploiting simple pixel-level color shortcuts.
    """)
    
    # 3x2 Grid for Ablation Statistics
    ablation_data = [
        {"exp": "Exp 38 vs 36", "desc": "Pure Discrete Rotation", "base": "34.40%", "jitter": "51.21%", "gain": "+16.81 pp"},
        {"exp": "Exp 39 vs 35", "desc": "Weak Spatial Baseline", "base": "59.22%", "jitter": "80.53%", "gain": "+21.31 pp"},
        {"exp": "Exp 40 vs 9",  "desc": "Crop + Gaussian Blur", "base": "63.01%", "jitter": "80.65%", "gain": "+17.64 pp"},
        {"exp": "Exp 41 vs 13", "desc": "Crop + Flip + Blur", "base": "64.49%", "jitter": "84.30%", "gain": "+19.81 pp"},
        {"exp": "Exp 42 vs 10", "desc": "Crop + Random Cutout", "base": "66.27%", "jitter": "81.21%", "gain": "+14.94 pp"}
    ]
    
    cols_ablation = st.columns(5)
    for i, data in enumerate(ablation_data):
        with cols_ablation[i]:
            card_html = f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 15px; text-align: center;">
                <div style="color: #38BDF8; font-size: 1.1rem; font-weight: 700;">{data["exp"]}</div>
                <div style="color: #94A3B8; font-size: 0.85rem; height: 35px; overflow: hidden; margin-top: 4px;">{data["desc"]}</div>
                <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 8px 0;"/>
                <div style="color: #94A3B8; font-size: 0.8rem;">Without Jitter: <b>{data["base"]}</b></div>
                <div style="color: #10B981; font-size: 0.85rem; font-weight: 600; margin-top: 2px;">With Jitter: {data["jitter"]}</div>
                <div style="background-color: rgba(16, 185, 129, 0.15); color: #10B981; border-radius: 4px; padding: 2px 6px; font-size: 0.9rem; font-weight: 700; margin-top: 8px; display: inline-block;">
                    {data["gain"]} Gain
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("#### 🔑 Key Project Takeaways")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ##### 1. The Shortcut-Learning Problem 🧩
        Without Color Jitter, contrastive self-supervised encoders exploit low-level **color histograms** as a shortcut to maximize mutual information, rather than learning general shapes and semantic features. This results in weaker downstream representations, showing a severe performance limit (e.g. baseline Crop+Flip+Blur achieves only **64.49%**).
        
        ##### 2. The Color Jitter Shield 🛡️
        Adding photometric distortion (color jittering) forces the model to ignore color profiles and focus on invariant structures, spatial boundaries, and contours. This single ablation yields a massive average boost of **+18.1 pp** across all settings, pushing our best encoder (Exp 41) to a stellar **84.30% Top-1 accuracy**!
        """)
        
    with col2:
        st.markdown("""
        ##### 3. Model Architecture & Stem Tuning ⚙️
        Modifying the standard ResNet-50 conv1 stem from $7\\times7$ (stride 2) to a custom $3\\times3$ (stride 1) and removing the initial MaxPool was crucial to preserve the resolution of $32\\times32$ CIFAR-10 images. 
        
        ##### 4. Near Foundation-Model Upper Bound 🏆
        Our zero-shot CLIP ViT-B/32 foundation model evaluation sets the academic upper bound at **88.80%**. Our custom-trained SimCLR ResNet-50 achieves **95% of this performance** (**84.30%**) while using **8,000x less training data**!
        """)

    st.success("🎉 Weights & Biases Dashboard has archived all 5 experiments complete with training curves, checkpoints, and t-SNE files.")
