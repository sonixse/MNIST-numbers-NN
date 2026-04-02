"""Visualization and helper utilities."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


FIG_TITLE_SIZE = 9
SUBPLOT_TITLE_SIZE = 6
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7


def build_feature_map_figure(
    feature_maps: torch.Tensor,
    title: str,
    max_maps: int = 12,
) -> plt.Figure:
    fmap = feature_maps[0].detach().cpu()
    n_maps = min(max_maps, fmap.shape[0])
    cols = 4
    rows = int(np.ceil(n_maps / cols))
    vmin = float(fmap.min().item())
    vmax = float(fmap.max().item())

    fig, axes = plt.subplots(rows, cols, figsize=(3.0, rows * 0.95))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n_maps:
            ax.imshow(
                fmap[i],
                cmap="magma",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Map {i}", fontsize=SUBPLOT_TITLE_SIZE, pad=2.0)
        ax.axis("off")
    fig.suptitle(title, fontsize=FIG_TITLE_SIZE, fontweight="bold")
    fig.subplots_adjust(hspace=0.14, wspace=0.04, top=0.9, bottom=0.05, left=0.05, right=0.95)
    return fig


def build_filter_figure(model: torch.nn.Module, max_filters: int = 16) -> plt.Figure:
    weights = model.conv1.weight.data.cpu()
    n_filters = min(max_filters, weights.shape[0])
    cols = 4
    rows = int(np.ceil(n_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.0, rows * 0.72))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n_filters:
            ax.imshow(weights[i][0], cmap="gray", interpolation="nearest")
            ax.set_title(f"F{i}", fontsize=SUBPLOT_TITLE_SIZE, pad=1.0)
        ax.axis("off")
    fig.suptitle("Learned Convolution Filters", fontsize=FIG_TITLE_SIZE, fontweight="bold")
    fig.subplots_adjust(hspace=0.22, wspace=0.04, top=0.9, bottom=0.05, left=0.05, right=0.95)
    return fig


def build_predictions_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    max_items: int = 9,
) -> plt.Figure:
    n_items = min(max_items, images.shape[0])
    cols = 3
    rows = int(np.ceil(n_items / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.0, 3.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n_items:
            ax.imshow(images[i].squeeze(), cmap="gray")
            color = "green" if int(labels[i]) == int(preds[i]) else "crimson"
            ax.set_title(f"T:{int(labels[i])} P:{int(preds[i])}", color=color, fontsize=SUBPLOT_TITLE_SIZE, pad=1.0)
        ax.axis("off")
    fig.suptitle("Sample Predictions", fontsize=FIG_TITLE_SIZE, fontweight="bold")
    fig.subplots_adjust(hspace=0.16, wspace=0.04, top=0.9, bottom=0.05, left=0.05, right=0.95)
    fig.text(0.5, 0.01, "Green labels are correct, red labels are mistakes.", ha="center", fontsize=SUBPLOT_TITLE_SIZE, color="#17263c")

    return fig


def build_confusion_matrix_figure(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> plt.Figure:
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    for t, p in zip(targets, preds):
        matrix[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    im = ax.imshow(matrix.numpy(), cmap="YlGnBu")
    ax.set_title("Confusion Matrix", fontsize=FIG_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("True", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(10):
        for j in range(10):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=SUBPLOT_TITLE_SIZE)

    fig.tight_layout()
    # add caption
    fig.text(0.5, 0.01, "Diagonal values indicate correct predictions", ha="center", fontsize=SUBPLOT_TITLE_SIZE, color="#17263c")
    return fig


def build_probability_bar_figure(
    probabilities: torch.Tensor,
    true_label: Optional[int] = None,
) -> plt.Figure:
    probs = probabilities.squeeze().numpy()
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    
    # Set light background and dark text
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    
    bars = ax.bar(range(10), probs, color="#5B7DB1")

    pred = int(np.argmax(probs))
    bars[pred].set_color("#D95F02")
    if true_label is not None:
        bars[int(true_label)].set_color("#1B9E77")

    ax.set_ylim(0, 1)
    ax.set_xticks(range(10))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, colors="#17263c")
    ax.set_xlabel("Digit", fontsize=AXIS_LABEL_SIZE, color="#17263c")
    ax.set_ylabel("Probability", fontsize=AXIS_LABEL_SIZE, color="#17263c")
    ax.set_title("Prediction Confidence", fontsize=FIG_TITLE_SIZE, fontweight="bold", color="#17263c")
    ax.spines["top"].set_color("#17263c")
    ax.spines["right"].set_color("#17263c")
    ax.spines["left"].set_color("#17263c")
    ax.spines["bottom"].set_color("#17263c")
    fig.tight_layout()
    fig.text(0.5, 0.01, "Confidence bars: distribution width shows model certainty.", ha="center", fontsize=SUBPLOT_TITLE_SIZE, color="#17263c")
    return fig
      