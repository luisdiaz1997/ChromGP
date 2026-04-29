import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.autonotebook import tqdm
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_3D(Z, X, title="3D Plot"):
    """
    Plot a 3D structure using Plotly.

    Args:
        Z: 3D coordinates (N, 3)
        X: Color values (N,)
        title: Title of the plot
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
        marker=dict(
            size=4,
            color=X.squeeze(),
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    )])
    fig.update_layout(title=title)
    fig.show()

def train_batched(optimizer, model, X, y, device, steps=200, batch_size=1000):
    """
    Train a model in batches.

    Args:
        optimizer: Optimizer for training
        model: Model to train
        X: Input data
        y: Target data
        device: Device to use (CPU/GPU)
        steps: Number of training steps
        batch_size: Batch size

    Returns:
        Losses and latent variables over training
    """
    Zs = []
    losses = []
    N = len(X)

    for _ in tqdm(range(steps)):
        idx = torch.multinomial(torch.ones(N), num_samples=batch_size, replacement=False)
        X_batch = X[idx].to(device)
        y_batch = y[:, idx].to(device)

        optimizer.zero_grad()
        pY, qZ, qU, pU = model(X_batch.squeeze())
        y_norm = y_batch - y_batch.mean(dim=1, keepdims=True)

        L1 = pY.log_prob(y_norm).sum() * (N / batch_size)
        L2 = torch.distributions.kl_divergence(qU, pU).sum()
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        Zs.append(qZ.mean.T.detach().cpu().clone().numpy())

    return losses, Zs

def train(optimizer, model, X, y, device, steps=200):
    """
    Train a model with full-batch gradient descent.

    Args:
        optimizer: Optimizer for training
        model: Model to train
        X: Input data
        y: Target data
        device: Device to use (CPU/GPU)
        steps: Number of training steps

    Returns:
        Losses and latent variables over training
    """
    Zs = []
    losses = []
    N = len(X)

    X = X.to(device)
    y = y.to(device)

    for _ in tqdm(range(steps)):
        optimizer.zero_grad()
        pY, qZ, qU, pU = model(X.squeeze())
        y_norm = y - y.mean(dim=1, keepdims=True)

        L1 = pY.log_prob(y_norm).sum()
        L2 = torch.distributions.kl_divergence(qU, pU).sum()
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        Zs.append(qZ.mean.T.detach().cpu().clone().numpy())

    X = X.cpu()
    y = y.cpu()

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return losses, Zs