import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import get_simclr_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimCLREncoderOnly(nn.Module):
    def __init__(self, simclr_model, repr_dim=2048):
        super().__init__()
        self.simclr_model = simclr_model
        self.repr_dim = getattr(simclr_model, 'repr_dim', repr_dim)

    def forward(self, x):
        h, z = self.simclr_model(x)
        return h

def extract_features(encoder, loader, device):
    encoder.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = encoder(imgs.to(device))
            feats = out[0] if isinstance(out, tuple) else out
            all_feats.append(feats.cpu())
            all_labels.append(lbls)
    return torch.cat(all_feats), torch.cat(all_labels)

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp41_ckpt = os.path.join(base_dir, "All Experiment SimCLR 18 May 2026", "v12_flipblur_jitter_exp41", "outputs", "resnet50_exp41_ntxent_τ0.5_bs512_ep200_20260519_141524", "simclr_encoder_final.pth")
    
    if not os.path.exists(exp41_ckpt):
        print(f"Error: Checkpoint not found at {exp41_ckpt}")
        return

    print("Loading dataset...")
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD  = [0.2023, 0.1994, 0.2010]
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    data_dir = os.path.join(base_dir, 'data')
    plain_train = DataLoader(CIFAR10(data_dir, train=True, download=True, transform=test_transform), batch_size=512, shuffle=False)
    plain_test_ds = CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    plain_test = DataLoader(plain_test_ds, batch_size=512, shuffle=False)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Loading model...")
    model = get_simclr_model(backbone='resnet50', projection_dim=128)
    ckpt = torch.load(exp41_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    FEATURE_DIM = getattr(model, 'repr_dim', 2048)
    encoder = SimCLREncoderOnly(model, repr_dim=FEATURE_DIM).to(device)

    print("Extracting features...")
    train_feats, train_labels = extract_features(encoder, plain_train, device)
    test_feats, test_labels = extract_features(encoder, plain_test, device)

    probe_train_ds = TensorDataset(train_feats, train_labels)
    probe_test_ds  = TensorDataset(test_feats,  test_labels)

    probe_train_loader = DataLoader(probe_train_ds, batch_size=512, shuffle=True)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=512, shuffle=False)

    probe = nn.Linear(FEATURE_DIM, 10).to(device)
    probe_criterion = nn.CrossEntropyLoss()
    probe_optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Training linear probe (20 epochs)...")
    for ep in range(1, 21):
        probe.train()
        for feats, lbls in probe_train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            probe_optimizer.zero_grad()
            loss = probe_criterion(probe(feats), lbls)
            loss.backward()
            probe_optimizer.step()

    print("Evaluating and collecting predictions...")
    probe.eval()
    all_preds = []
    all_probs = []
    all_true = []
    with torch.no_grad():
        for feats, lbls in probe_test_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            out = probe(feats)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(1)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_true.append(lbls.cpu())
            
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_true = torch.cat(all_true)

    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_true.numpy(), all_preds.numpy())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (SimCLR Exp 41 Linear Probe)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(base_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # 2. OOD Errors (
    print("Generating OOD Errors...")
    errors_idx = (all_preds != all_true).nonzero(as_tuple=True)[0]
    
    # Getting the predicted probabilities for the misclassified examples
    err_probs = all_probs[errors_idx, all_preds[errors_idx]]
    
    sorted_err_idx = errors_idx[torch.argsort(err_probs, descending=True)]
    
    top_errors = sorted_err_idx[:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_errors):
        img_t, _ = plain_test_ds[idx]
        img_denorm = denormalize(img_t.clone(), CIFAR_MEAN, CIFAR_STD)
        img_np = np.clip(img_denorm.permute(1, 2, 0).numpy(), 0, 1)
        
        true_cls = classes[all_true[idx].item()]
        pred_cls = classes[all_preds[idx].item()]
        conf = all_probs[idx, all_preds[idx]].item() * 100
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)", color='red' if true_cls != pred_cls else 'black')
        axes[i].axis('off')
        
    plt.suptitle("High-Confidence Misclassifications (OOD Errors)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    ood_path = os.path.join(base_dir, 'ood_errors.png')
    plt.savefig(ood_path, dpi=300)
    plt.close()
    print(f"Saved OOD errors to {ood_path}")

if __name__ == "__main__":
    main()
