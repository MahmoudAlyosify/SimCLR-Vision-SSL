
  #Export SimCLR Encoder to ONNX


import os
import sys
import time
import numpy as np

# Windows console encoding fix (tau char in folder names)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torchvision.models as models


# ==========================================================================
# 1. Rebuild the encoder-only architecture (no projection head)
# ==========================================================================
class SimCLREncoder(nn.Module):
    """
    ResNet-50 encoder with CIFAR-10 stem modification.
    Outputs 2048-d feature vectors (no projection head).

    Architecture matches SimCLRResNet50.encoder + pool + flatten
    from model.py exactly, so we can load the same state_dict keys.
    """

    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=None)

        # CIFAR-10 stem: 3x3 conv, stride 1, no maxpool
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: (N, 3, 32, 32) CIFAR-10 images, normalized
        Returns:
            h: (N, 2048) feature embeddings
        """
        x = self.encoder(x)
        x = self.pool(x)
        h = x.flatten(start_dim=1)
        return h


# ==========================================================================
# 2. Configuration
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.join(BASE_DIR, "All Experiment SimCLR 18 May 2026")
EXP41_FOLDER = "v12_flipblur_jitter_exp41"

OUTPUT_DIR = os.path.join(BASE_DIR, "deployment")
ONNX_FILENAME = "simclr_encoder_exp41.onnx"

# CIFAR-10 input shape
INPUT_SHAPE = (1, 3, 32, 32)


def find_checkpoint():
    """Auto-discover the .pth checkpoint inside Exp 41's output folder."""
    outputs_dir = os.path.join(EXP_ROOT, EXP41_FOLDER, "outputs")
    if not os.path.isdir(outputs_dir):
        print(f"[ERROR] Outputs directory not found: {outputs_dir}")
        sys.exit(1)

    for run_dir in os.listdir(outputs_dir):
        pth = os.path.join(outputs_dir, run_dir, "simclr_encoder_final.pth")
        if os.path.exists(pth):
            return pth

    print(f"[ERROR] No simclr_encoder_final.pth found in {outputs_dir}")
    sys.exit(1)


def main():
    print("=" * 72)
    print("  SimCLR Encoder -> ONNX Export")
    print("  Experiment 41: Crop+Flip+Blur+Jitter (84.30% Top-1)")
    print("=" * 72)

    # -- Find checkpoint -------------------------------------------------
    ckpt_path = find_checkpoint()
    ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"\n[1/4] Checkpoint found:")
    print(f"      {ckpt_path}")
    print(f"      Size: {ckpt_size:.1f} MB")

    # -- Build encoder and load weights ----------------------------------
    print("\n[2/4] Building encoder and loading weights...")
    t0 = time.time()

    model = SimCLREncoder()
    full_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Filter: keep only encoder.* and pool.* keys (drop projection.*)
    encoder_keys = {
        k: v for k, v in full_state.items()
        if k.startswith("encoder.") or k.startswith("pool.")
    }
    projection_keys = [k for k in full_state if k.startswith("projection.")]

    model.load_state_dict(encoder_keys, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      Loaded {len(encoder_keys)} encoder keys ({n_params:.1f}M params)")
    print(f"      Dropped {len(projection_keys)} projection head keys")
    print(f"      Load time: {time.time() - t0:.1f}s")

    # -- Export to ONNX --------------------------------------------------
    print("\n[3/4] Exporting to ONNX...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)

    dummy_input = torch.randn(*INPUT_SHAPE)

    t1 = time.time()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={
            "images":   {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        dynamo=False,
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"      Exported: {onnx_path}")
    print(f"      ONNX size: {onnx_size:.1f} MB")
    print(f"      Export time: {time.time() - t1:.1f}s")

    # -- Validate ONNX vs PyTorch ----------------------------------------
    print("\n[4/4] Validating ONNX against PyTorch...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

        # PyTorch output
        with torch.no_grad():
            pt_out = model(torch.from_numpy(test_input)).numpy()

        # ONNX output
        onnx_out = session.run(None, {"images": test_input})[0]

        max_err = np.max(np.abs(pt_out - onnx_out))
        mean_err = np.mean(np.abs(pt_out - onnx_out))

        status = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"      Max  error: {max_err:.2e}")
        print(f"      Mean error: {mean_err:.2e}")
        print(f"      Validation: [{status}]")

        if status == "FAIL":
            print("      [WARNING] Error exceeds threshold. Check opset compatibility.")

    except ImportError:
        print("      [SKIP] onnxruntime not installed. Install with:")
        print("             pip install onnxruntime")
        print("      (ONNX file was exported successfully regardless.)")

    # -- Summary ---------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  ONNX EXPORT COMPLETE")
    print(f"  File   : {onnx_path}")
    print(f"  Size   : {onnx_size:.1f} MB")
    print(f"  Input  : images (batch, 3, 32, 32) float32")
    print(f"  Output : features (batch, 2048) float32")
    print(f"  Ready for FAISS indexing and GUI deployment!")
    print("=" * 72)


if __name__ == "__main__":
    main()
