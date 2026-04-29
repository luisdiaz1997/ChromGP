import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm.autonotebook import tqdm

def make_helix(num_points=1000, radius=1.0, turns=5):
    """
    Create a 3D helix using PyTorch.

    Args:
        num_points: Number of points in the helix
        radius: Radius of the circular component
        turns: Number of full 2π turns of the helix

    Returns:
        Tensor of shape (num_points, 3): the (x, y, z) coordinates
    """
    t = torch.linspace(0, 2 * torch.pi * turns, num_points)
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = t / (torch.pi * turns)  # Linear increase

    return torch.stack([x, y, z], dim=1)

def compute_contacts(simulations2D):
    """
    Convert a list of distance matrices into symmetric contact maps
    using a Poisson distance-decay model.

    Args:
        simulations2D: list of torch.Tensor of shape (N, N), pairwise distances

    Returns:
        contacts2D: list of torch.Tensor of shape (N, N), symmetric contact matrices
    """
    contacts2D = []

    for D in simulations2D:
        lam = 1 / (1 + D**2)                  # Step 1: distance-decay rate
        samples = torch.poisson(lam)          # Step 2: Poisson sampling
        contact_matrix = torch.tril(samples) + torch.tril(samples, -1).T  # Step 3: symmetrize
        contacts2D.append(contact_matrix)     # Step 4: collect

    return contacts2D



def generate_simulations(Z, num_simulations=16, noise_level=0.15):
    """
    Generate noisy simulations of a 3D structure.

    Args:
        Z: Original 3D structure (N, 3)
        num_simulations: Number of simulations to generate
        noise_level: Standard deviation of the noise

    Returns:
        List of noisy 3D structures and their pairwise distances
    """
    simulationsZ = []
    simulations2D = []

    for _ in range(num_simulations):
        Z_noise = Z + noise_level * torch.randn(Z.shape)
        simulationsZ.append(Z_noise)
        Z_dist = torch.cdist(Z_noise, Z_noise)
        simulations2D.append(Z_dist)

    return simulationsZ, simulations2D

def create_animation(Zs, Z, output_file="animation.mp4", step=20, fps=50, interval=100):
    """
    Create an animation of 3D structures and distances.

    Args:
        Zs: List of 3D structures over time
        Z: Ground truth 3D structure
        output_file: Path to save the animation
        interval: Interval between frames in milliseconds
    """
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')  # 3D plot
    ax2 = fig.add_subplot(1, 3, 2)                   # Matshow plot 1
    ax3 = fig.add_subplot(1, 3, 3)                   # Matshow plot 2

    titles = ["3D coordinates", "Reconstructed distances", "True distances"]
    pbar = tqdm(total=len(np.arange(0, len(Zs), step)), desc="Animating")

    def update(iteration):
        ax1.cla()
        curr_Z = Zs[iteration]

        ax1.plot(curr_Z[:, 0], curr_Z[:, 1], curr_Z[:, 2], label='3D structure', lw=1.0)
        ax1.legend()
        ax1.view_init(elev=20, azim=-100)
        ax1.set_title(f"{titles[0]} (step {iteration})")

        ax2.cla()
        ax3.cla()
        reconstructed_dist = torch.cdist(curr_Z, curr_Z)
        ax2.matshow(reconstructed_dist, cmap='YlOrRd_r')
        ax2.set_title(titles[1])
        ax3.matshow(torch.cdist(Z, Z), cmap='YlOrRd_r')
        ax3.set_title(titles[2])

        pbar.update(1)

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(Zs), step), interval=interval)
    anim.save(output_file, fps=fps, dpi=100)
    pbar.close()
