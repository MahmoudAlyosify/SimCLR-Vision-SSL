import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):

        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.shape[0]

        device = z1.device

        # normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # combine both views
        representations = torch.cat(
            [z1, z2],
            dim=0
        )

        # similarity matrix
        similarity_matrix = torch.mm(
            representations,
            representations.T
        )

        similarity_matrix = similarity_matrix / self.temperature

        # remove diagonal values
        diagonal_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=device
        )

        similarity_matrix = similarity_matrix.masked_fill(
            diagonal_mask,
            float("-inf")
        )

        # positive pair targets
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])

        loss = F.cross_entropy(
            similarity_matrix,
            targets
        )

        return loss


# =========================================================
# Simple Test
# =========================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing NT-Xent Loss")
    print("=" * 50)

    criterion = NTXentLoss(temperature=0.5)

    batch_size = 32
    feature_dim = 128

    # random embeddings
    z1 = torch.randn(batch_size, feature_dim)
    z2 = torch.randn(batch_size, feature_dim)

    loss = criterion(z1, z2)

    expected_loss = torch.log(
        torch.tensor(2 * batch_size - 1.0)
    )

    print("\nTest 1 -> Random embeddings")
    print(f"Loss value : {loss.item():.4f}")
    print(f"Expected   : {expected_loss:.4f}")

    # identical embeddings
    perfect_loss = criterion(
        z1,
        z1.clone()
    )

    print("\nTest 2 -> Identical embeddings")
    print(f"Loss value : {perfect_loss.item():.6f}")

    # scalar output check
    assert loss.shape == torch.Size([])

    print("\nTest 3 -> Scalar output passed")

    print("=" * 50)