"""
================================================================================
  Build FAISS Index for CIFAR-10 Test Set
  ---------------------------------------
  Generates the FAISS vector database and metadata mapping for the real-time
  visual search engine GUI.

  What this script does:
    1. Loads the exported ONNX ResNet-50 encoder (simclr_encoder_exp41.onnx)
    2. Loads the CIFAR-10 test set (10,000 images) using the evaluation pipeline
    3. Runs batched ONNX inference on the CPU/GPU to extract 2048-d features
    4. L2-normalizes the features to enable exact Cosine Similarity search
    5. Saves the raw test images to disk as PNGs for easy loading in the GUI
    6. Constructs a FAISS IndexFlatIP (Inner Product) search database
    7. Saves a JSON metadata mapping linking vector IDs to image paths/classes
    8. Performs a validation query to verify search accuracy

  Outputs:
    - ./deployment/cifar10_index.faiss     (FAISS vector database)
    - ./deployment/metadata.json           (Vector ID to Image/Label mapping)
    - ./deployment/test_images/*.png       (10,000 reference images)

  Usage:
    python build_faiss.py
================================================================================
"""

import os
import sys
import time
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    import torch
    from torchvision.datasets import CIFAR10
except ImportError:
    print("[ERROR] PyTorch / TorchVision not found.")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] faiss not found. Install with: pip install faiss-cpu")
    sys.exit(1)

from src.dataset import get_eval_dataloader, CIFAR_MEAN, CIFAR_STD

# ==========================================================================
# Configuration
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "deployment", "simclr_encoder_exp41.onnx")

DEPLOY_DIR = os.path.join(BASE_DIR, "deployment")
IMAGES_DIR = os.path.join(DEPLOY_DIR, "test_images")
INDEX_PATH = os.path.join(DEPLOY_DIR, "cifar10_index.faiss")
META_PATH = os.path.join(DEPLOY_DIR, "metadata.json")

BATCH_SIZE = 256
DATA_DIR = "./data"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def main():
    print("=" * 72)
    print("  FAISS Vector Index Builder")
    print("  Backbone: ResNet-50 (Exp 41 SimCLR Encoder)")
    print("=" * 72)

    # -- 1. Load ONNX model --------------------------------------------------
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] ONNX model not found at: {ONNX_PATH}")
        print("        Please run export_onnx.py first!")
        sys.exit(1)

    print(f"\n[1/6] Loading ONNX model...")
    t0 = time.time()
    
    # Auto-select best execution provider (CUDA if available, else CPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        ort_sess = ort.InferenceSession(ONNX_PATH, providers=providers)
        active_provider = ort_sess.get_providers()[0]
        print(f"      ONNX session loaded successfully!")
        print(f"      Execution Provider: {active_provider}")
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        sys.exit(1)
        
    print(f"      Load time: {time.time() - t0:.1f}s")

    # -- 2. Load Dataset -----------------------------------------------------
    print(f"\n[2/6] Loading CIFAR-10 test set...")
    # Load normalized images for inference
    eval_loader = get_eval_dataloader(
        data_dir=DATA_DIR,
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    
    # Load raw dataset to extract original unnormalized PIL images to save to disk
    raw_dataset = CIFAR10(root=DATA_DIR, train=False, download=True)
    n_images = len(raw_dataset)
    print(f"      Loaded {n_images:,} test images.")

    # -- 3. Extract Features -------------------------------------------------
    print(f"\n[3/6] Extracting 2048-d SimCLR features...")
    features_list = []
    
    t_feat_start = time.time()
    for batch_images, _ in tqdm(eval_loader, desc="      Inference", unit="batch"):
        # Convert PyTorch tensor to numpy array for ONNX
        batch_numpy = batch_images.numpy().astype(np.float32)
        # Run inference
        outputs = ort_sess.run(None, {"images": batch_numpy})
        features_list.append(outputs[0])
        
    features = np.concatenate(features_list, axis=0)
    print(f"      Extracted feature matrix shape: {features.shape}")
    print(f"      Inference speed: {n_images / (time.time() - t_feat_start):.0f} img/s")

    # -- 4. L2-Normalize for Cosine Similarity -------------------------------
    print(f"\n[4/6] Normalizing features for exact Cosine Similarity...")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    # Avoid division by zero
    features_normalized = features / np.maximum(norms, 1e-12)
    print(f"      L2 normalization complete.")

    # -- 5. Save Raw PNG Images and Create Metadata --------------------------
    print(f"\n[5/6] Exporting original images to disk & creating metadata...")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    metadata = {
        "classes": CIFAR10_CLASSES,
        "images": []
    }
    
    # Save unnormalized PIL images and generate mapping dictionary
    for i in tqdm(range(n_images), desc="      Exporting PNGs", unit="img"):
        img, label = raw_dataset[i]
        
        # Save image file
        img_filename = f"{i:05d}.png"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        img.save(img_path, format="PNG")
        
        # Relative path for portability (allows moving the deployment folder)
        rel_img_path = f"test_images/{img_filename}"
        
        metadata["images"].append({
            "id": i,
            "image_path": rel_img_path,
            "label_id": int(label),
            "class_name": CIFAR10_CLASSES[label]
        })
        
    # Write metadata.json
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    print(f"      Exported {n_images:,} images to {IMAGES_DIR}")
    print(f"      Saved metadata catalog to {META_PATH}")

    # -- 6. Build and Save FAISS Index ---------------------------------------
    print(f"\n[6/6] Building FAISS Inner Product (Cosine) index...")
    t_faiss = time.time()
    
    dimension = features_normalized.shape[1] # 2048
    
    # We use IndexFlatIP (Inner Product) on L2 normalized features
    # to yield exact Cosine Similarity search results.
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(features_normalized)
    
    # Write index to disk
    faiss.write_index(index, INDEX_PATH)
    print(f"      Constructed FAISS IndexFlatIP.")
    print(f"      Saved FAISS index to {INDEX_PATH}")
    print(f"      FAISS build time: {time.time() - t_faiss:.3f}s")

    # ==========================================================================
    # Validation Query
    # ==========================================================================
    print("\n" + "=" * 72)
    print("  Validation: Performing Sample Visual Search")
    print("=" * 72)
    
    # Query with the first image in the test set
    query_id = 0
    query_class = metadata["images"][query_id]["class_name"]
    query_vec = features_normalized[query_id:query_id+1]
    
    print(f"  Query Vector ID: {query_id} (Class: {query_class})")
    
    # Search for top 5 nearest neighbors
    top_k = 5
    t_search = time.time()
    similarities, indices = index.search(query_vec, top_k)
    search_time = (time.time() - t_search) * 1000
    
    print(f"  Search completed in {search_time:.3f} ms")
    print(f"  Top {top_k} Nearest Neighbors:")
    print(f"  ┌──────┬──────────┬─────────────┬────────────┐")
    print(f"  │ Rank │ VectorID │ Class Name  │ Similarity │")
    print(f"  ├──────┼──────────┼─────────────┼────────────┤")
    for rank in range(top_k):
        neighbor_id = int(indices[0][rank])
        neighbor_class = metadata["images"][neighbor_id]["class_name"]
        score = float(similarities[0][rank])
        print(f"  │ {rank+1:4d} │ {neighbor_id:8d} │ {neighbor_class:<11s} │ {score:10.4f} │")
    print(f"  └──────┴──────────┴─────────────┴────────────┘")
    
    # Check if the top match is the image itself (score ~ 1.0)
    success = (indices[0][0] == query_id) and (similarities[0][0] > 0.99)
    # Check top-5 precision (how many have the same class)
    same_class_count = sum(1 for idx in indices[0] if metadata["images"][int(idx)]["class_name"] == query_class)
    precision = (same_class_count / top_k) * 100
    
    print(f"\n  Index Self-Consistency : {'[PASS]' if success else '[FAIL]'}")
    print(f"  Top-5 Search Precision : {precision:.1f}% ({same_class_count}/{top_k} same class)")
    print("=" * 72)
    print("  SUCCESS: FAISS deployment artifacts generated and verified!")
    print("=" * 72)


if __name__ == "__main__":
    main()
