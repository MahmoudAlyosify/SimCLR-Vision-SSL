import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    checkpoints = [
        (36, "v2_pure_rotation", r"v2_pure_rotation\outputs\resnet50_exp36_ntxent_τ0.5_bs512_ep200_20260517_153808\simclr_encoder_final.pth"),
        (35, "v3_weak_baseline", r"v3_weak_baseline\outputs\resnet50_exp35_ntxent_τ0.5_bs512_ep200_20260517_174004\simclr_encoder_final.pth"),
        (9, "v4_crop_blur", r"v4_crop_blur\outputs\resnet50_exp9_ntxent_τ0.5_bs512_ep200_20260517_194204\simclr_encoder_final.pth"),
        (13, "v5_crop_flip_blur", r"v5_crop_flip_blur\outputs\resnet50_exp13_ntxent_τ0.5_bs512_ep200_20260517_214535\simclr_encoder_final.pth"),
        (10, "v6_crop_cutout", r"v6_crop_cutout\outputs\resnet50_exp10_ntxent_τ0.5_bs512_ep200_20260517_234848\simclr_encoder_final.pth")
    ]

    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD  = [0.2023, 0.1994, 0.2010]
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    plain_train = DataLoader(
        CIFAR10(data_dir, train=True, download=True, transform=test_transform),
        batch_size=512, shuffle=False, num_workers=0, pin_memory=False
    )
    plain_test = DataLoader(
        CIFAR10(data_dir, train=False, download=True, transform=test_transform),
        batch_size=512, shuffle=False, num_workers=0, pin_memory=False
    )

    results = []

    for exp_id, folder, ckpt_rel in checkpoints:
        ckpt_path = os.path.join(base_dir, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"File not found: {ckpt_path}")
            continue
            
        print(f"\n_______________________________________________________")
        print(f"Evaluating Experiment {exp_id} ({folder})")
        print(f"_______________________________________________________")
        
        # Loading our model
        model = get_simclr_model(backbone='resnet50', projection_dim=128)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        model.eval()
        FEATURE_DIM = getattr(model, 'repr_dim', 2048)
        encoder = SimCLREncoderOnly(model, repr_dim=FEATURE_DIM).to(device)
        
        print('Extracting train features...')
        train_feats, train_labels = extract_features(encoder, plain_train, device)
        print('Extracting test features...')
        test_feats, test_labels = extract_features(encoder, plain_test, device)
        
        probe_train_ds = TensorDataset(train_feats, train_labels)
        probe_test_ds  = TensorDataset(test_feats,  test_labels)

        probe_train_loader = DataLoader(probe_train_ds, batch_size=512, shuffle=True)
        probe_test_loader  = DataLoader(probe_test_ds,  batch_size=512, shuffle=False)

        probe = nn.Linear(FEATURE_DIM, 10).to(device)
        probe_criterion = nn.CrossEntropyLoss()
        probe_optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

        PROBE_EPOCHS = 50
        print(f"{'Epoch':>6}  {'Train Acc':>9}  {'Test Acc':>8}")
        print('-' * 30)

        for ep in range(1, PROBE_EPOCHS + 1):
            probe.train()
            correct, total = 0, 0
            for feats, lbls in probe_train_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                probe_optimizer.zero_grad()
                out  = probe(feats)
                loss = probe_criterion(out, lbls)
                loss.backward()
                probe_optimizer.step()
                correct += out.argmax(1).eq(lbls).sum().item()
                total   += lbls.size(0)
            tr_acc = 100.0 * correct / total
            
            probe.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for feats, lbls in probe_test_loader:
                    feats, lbls = feats.to(device), lbls.to(device)
                    out = probe(feats)
                    correct += out.argmax(1).eq(lbls).sum().item()
                    total   += lbls.size(0)
            te_acc = 100.0 * correct / total
            
            if ep % 10 == 0 or ep == 1:
                print(f"{ep:>6}  {tr_acc:>8.2f}%  {te_acc:>7.2f}%")

        probe_final_acc = te_acc
        print('_' * 30)
        print(f'Linear Probe Final Test Acc: {probe_final_acc:.2f}%')
        results.append((exp_id, folder, probe_final_acc))

    print("\n" + "_"*60)
    print(" FINAL RESULTS SUMMARY (LINEAR PROBE - RESNET50) ")
    print("_"*60)
    for r in results:
        print(f"Exp {r[0]:<2} | {r[1]:<20} | Top-1 Test Acc: {r[2]:.2f}%")
        
if __name__ == '__main__':
    main()
