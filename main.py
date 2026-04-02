import argparse
import os

import torch
import torch.optim as optim

from gpu.gpu import get_device
from model.mnist_cnn import MNISTNet, evaluate, get_dataloaders, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST CNN training entrypoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-path", type=str, default="output/mnist_cnn.pt")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        test_batch_size=1000,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )

    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, device, train_loader, optimizer)
        eval_stats = evaluate(model, device, test_loader)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy'] * 100:.2f}% "
            f"test_loss={float(eval_stats['loss']):.4f} "
            f"test_acc={float(eval_stats['accuracy']) * 100:.2f}%"
        )

    checkpoint_name = os.path.basename(args.save_path) or "mnist_cnn.pt"
    final_save_path = os.path.join("output", checkpoint_name)
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), final_save_path)
    print(f"Model saved to {final_save_path}")


if __name__ == "__main__":
    run()