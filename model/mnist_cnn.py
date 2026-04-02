"""Model for MNIST CNN components."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        x = F.max_pool2d(conv2_out, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        log_probs = F.log_softmax(logits, dim=1)

        if return_features:
            return log_probs, conv1_out, conv2_out
        return log_probs


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms()
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if limit_train is not None:
        train_set = Subset(train_set, list(range(min(limit_train, len(train_set)))))
    if limit_test is not None:
        test_set = Subset(test_set, list(range(min(limit_test, len(test_set)))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def evaluate(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
) -> Dict[str, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    first_batch = None

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction="sum")

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item()

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

            if batch_idx == 0:
                first_batch = (data.cpu(), target.cpu(), pred.cpu())

    return {
        "loss": torch.tensor(total_loss / max(total, 1), dtype=torch.float32),
        "accuracy": torch.tensor(correct / max(total, 1), dtype=torch.float32),
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
        "sample_batch": first_batch,
    }


def predict_single(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        log_probs, conv1_out, conv2_out = model(image_tensor, return_features=True)
        probs = torch.exp(log_probs)
        pred = probs.argmax(dim=1)

    return {
        "prediction": pred.cpu(),
        "probabilities": probs.cpu(),
        "conv1": conv1_out.cpu(),
        "conv2": conv2_out.cpu(),
    }