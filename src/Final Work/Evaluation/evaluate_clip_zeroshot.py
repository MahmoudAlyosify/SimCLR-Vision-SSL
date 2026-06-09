import sys
import time

#This is just a windows encoding fix
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

#Dependency check
try:
    import torch
    import torchvision
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
except ImportError:
    print("[ERROR] PyTorch / TorchVision not found.")
    sys.exit(1)

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("[ERROR] Hugging Face Transformers not found.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] tqdm not found.")
    sys.exit(1)


#Our configuration
MODEL_ID   = "openai/clip-vit-base-patch32"   
BATCH_SIZE = 256
NUM_WORKERS = 4
DATA_ROOT  = "./data"                      

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

PROMPT_TEMPLATE = "a photo of a {}"

#Building the zero-shot text prompts
def build_text_prompts(class_names: list[str]) -> list[str]:
    return [PROMPT_TEMPLATE.format(name) for name in class_names]

#This function is to keep images as raw PIL objects
def pil_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print("_" * 72)
    print("  CLIP Zero-Shot Evaluation on CIFAR-10")
    print("_" * 72)
    print(f"  Model     : {MODEL_ID}")
    print(f"  Device    : {device} ({gpu_name})")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Workers   : {NUM_WORKERS}")
    print("─" * 72)

    print("\n[1/3] Loading CLIP model and processor...")
    t0 = time.time()
    
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      Model loaded in {time.time() - t0:.1f}s  "
          f"({num_params:.1f}M parameters)")

    #Preparing text embeddings
    text_prompts = build_text_prompts(CIFAR10_CLASSES)
    print(f"\n[2/3] Text prompts ({len(text_prompts)} classes):")
    for i, prompt in enumerate(text_prompts):
        print(f"       [{i}] \"{prompt}\"")
    
    test_dataset = CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=None,      
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pil_collate_fn,
    )
    
    print(f"      {len(test_dataset):,} test images, "
          f"{len(test_loader)} batches of {BATCH_SIZE}")

    #Zero-Shot Inference Loop

    print("\n" + "─" * 72)
    print("  Running Zero-Shot Inference...")
    print("─" * 72)
    
    correct = 0
    total   = 0
    per_class_correct = [0] * len(CIFAR10_CLASSES)
    per_class_total   = [0] * len(CIFAR10_CLASSES)
    
    t_start = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating",
                                    unit="batch", ncols=80):
            #Process batch: images (PIL) + text prompts → tensors
            inputs = processor(
                text=text_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            #Moving all tensors to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            logits = outputs.logits_per_image       
            probs  = logits.softmax(dim=-1)          
            preds  = probs.argmax(dim=-1).cpu()      
            
            # Accumulate metrics
            labels_cpu = labels
            correct += (preds == labels_cpu).sum().item()
            total   += labels_cpu.size(0)
            
            for pred, gt in zip(preds, labels_cpu):
                per_class_total[gt.item()] += 1
                if pred.item() == gt.item():
                    per_class_correct[gt.item()] += 1
    
    elapsed = time.time() - t_start
    accuracy = 100.0 * correct / total
    

    # Results Summary

    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + "  CLIP Zero-Shot Results (CIFAR-10 Test Set)".center(70) + "║")
    print("╠" + "═" * 70 + "╣")
    print("║" + f"  Model        : {MODEL_ID}".ljust(70) + "║")
    print("║" + f"  Device       : {gpu_name}".ljust(70) + "║")
    print("║" + f"  Test Images  : {total:,}".ljust(70) + "║")
    print("║" + f"  Inference    : {elapsed:.1f}s  "
          f"({total/elapsed:.0f} img/s)".ljust(70) + "║")
    print("╠" + "═" * 70 + "╣")
    print("║" + f"  ★  Top-1 Zero-Shot Accuracy:  {accuracy:.2f}%  "
          f"({correct}/{total})  ★".center(70) + "║")
    print("╠" + "═" * 70 + "╣")
    print("║" + "  Per-Class Breakdown:".ljust(70) + "║")
    print("║" + "  " + "─" * 50 + " " * 18 + "║")
    
    for i, cls_name in enumerate(CIFAR10_CLASSES):
        cls_acc = 100.0 * per_class_correct[i] / max(per_class_total[i], 1)
        bar_len = int(cls_acc / 2.5)   # scale to ~40 chars max
        bar = "█" * bar_len
        line = f"  {cls_name:<12s}  {cls_acc:5.1f}%  ({per_class_correct[i]:4d}/{per_class_total[i]:4d})  {bar}"
        print("║" + line.ljust(70) + "║")
    
    print("╚" + "═" * 70 + "╝")
    
    
    print("\n  Context (CIFAR-10 Top-1 Accuracy):")
    print("  ┌────────────────────────────────────────────────────┐")
    print(f"  │  Our SimCLR (ResNet-50, 200ep) :  84.30%           │")
    print(f"  │  CLIP ViT-B/32 (Zero-Shot)    :  {accuracy:.2f}%           │")
    print(f"  │  Supervised CE (ResNet-50)     :  93.77%           │")
    print("  └────────────────────────────────────────────────────┘")
    
    return accuracy


if __name__ == "__main__":
    accuracy = main()
