import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def evaluate_model(ckpt_path, plain_train, plain_test):
    safe_path = str(ckpt_path).encode('ascii', 'replace').decode('ascii')
    print(f"\n___ Evaluating: {safe_path} ___")
    model = get_simclr_model(backbone='resnet50', projection_dim=128)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    FEATURE_DIM = getattr(model, 'repr_dim', 2048)
    encoder = SimCLREncoderOnly(model, repr_dim=FEATURE_DIM).to(device)
    
    train_feats, train_labels = extract_features(encoder, plain_train, device)
    test_feats, test_labels = extract_features(encoder, plain_test, device)
    
    probe_train_ds = TensorDataset(train_feats, train_labels)
    probe_test_ds  = TensorDataset(test_feats,  test_labels)

    probe_train_loader = DataLoader(probe_train_ds, batch_size=512, shuffle=True)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=512, shuffle=False)

    probe = nn.Linear(FEATURE_DIM, 10).to(device)
    probe_criterion = nn.CrossEntropyLoss()
    probe_optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, 51):
        probe.train()
        for feats, lbls in probe_train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            probe_optimizer.zero_grad()
            out  = probe(feats)
            loss = probe_criterion(out, lbls)
            loss.backward()
            probe_optimizer.step()
            
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, lbls in probe_test_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            out = probe(feats)
            correct += out.argmax(1).eq(lbls).sum().item()
            total   += lbls.size(0)
    te_acc = 100.0 * correct / total
    print(f"Final Test Acc: {te_acc:.2f}%")
    return te_acc

def generate_plot(results_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titlesize': 16,
        'figure.dpi': 300
    })

    experiments = [
        "Exp 36\n(Pure Rotation)",
        "Exp 35\n(Weak Baseline)",
        "Exp 9\n(Crop + Blur)",
        "Exp 13\n(Crop+Flip+Blur)",
        "Exp 10\n(Crop + Cutout)"
    ]

    acc_without_jitter = [34.40, 59.22, 63.01, 64.49, 66.27]
    
    acc_with_jitter = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    #Mapping the folder names to the correct index in the acc_with_jitter array
    folder_to_index = {
        "v9_rotation_jitter_exp38": 0,
        "v10_weakbaseline_jitter_exp39": 1,
        "v11_blur_jitter_exp40": 2,
        "v12_flipblur_jitter_exp41": 3,
        "v13_cutout_jitter_exp42": 4,

        # Keeping the old ones as they are for backwards compatibility
        "v7_baseline_with_jitter": 1,
        "v8_ultimate_beast": 4
    }
    
    for folder_name, idx in folder_to_index.items():
        if folder_name in results_dict:
            acc_with_jitter[idx] = results_dict[folder_name]

    df = pd.DataFrame({
        'Experiment': experiments * 2,
        'Top-1 Accuracy (%)': acc_without_jitter + acc_with_jitter,
        'Condition': ['Base Augmentation\n(Without Color Jitter)'] * 5 + ['Base + Color Jitter\n(With Jitter)'] * 5
    })

    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = {"Base Augmentation\n(Without Color Jitter)": "#1f77b4", "Base + Color Jitter\n(With Jitter)": "#2ca02c"}
    
    barplot = sns.barplot(
        data=df, 
        x='Experiment', 
        y='Top-1 Accuracy (%)', 
        hue='Condition',
        palette=palette,
        edgecolor='black',
        linewidth=1.2,
        ax=ax
    )

    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=11, fontweight='bold')

    ax.set_title('Impact of Color Jitter on Shortcut Learning in SimCLR\n(Linear Probe Top-1 Test Accuracy)', pad=20, fontweight='bold')
    ax.set_ylabel('Top-1 Test Accuracy (%)', fontweight='bold', labelpad=15)
    ax.set_xlabel('Structural Augmentation Strategy', fontweight='bold', labelpad=15)
    ax.set_ylim(0, 100)
    ax.legend(title='', loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9, edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    out_png = os.path.join(output_dir, 'ablation_shortcut_learning_final.png')
    out_pdf = os.path.join(output_dir, 'ablation_shortcut_learning_final.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_and_plot_beast.py <pipeline_dir>")
        sys.exit(1)
        
    pipeline_dir = sys.argv[1]
    print(f"Scanning for models in {pipeline_dir}...")
    
    #Setting up the data loaders
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD  = [0.2023, 0.1994, 0.2010]
    test_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    plain_train = DataLoader(CIFAR10(data_dir, train=True, download=True, transform=test_transform), batch_size=512, shuffle=False, num_workers=0)
    plain_test = DataLoader(CIFAR10(data_dir, train=False, download=True, transform=test_transform), batch_size=512, shuffle=False, num_workers=0)

    results_dict = {}
    
    #Find all simclr_encoder_final.pth
    for root, dirs, files in os.walk(pipeline_dir):
        if "simclr_encoder_final.pth" in files:
            ckpt_path = os.path.join(root, "simclr_encoder_final.pth")
            #Extracting folder name from path
            parts = Path(ckpt_path).parts
            try:
                exp_folder = parts[-4]
                acc = evaluate_model(ckpt_path, plain_train, plain_test)
                results_dict[exp_folder] = acc
            except Exception as e:
                safe_err_path = str(ckpt_path).encode('ascii', 'replace').decode('ascii')
                print(f"Error evaluating {safe_err_path}: {e}")

    results_dir = os.path.join(pipeline_dir, "Results")
    generate_plot(results_dict, results_dir)

if __name__ == '__main__':
    from pathlib import Path
    main()
