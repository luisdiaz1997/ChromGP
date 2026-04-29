from torch import nn
import torch
import torch.distributions as distributions
from gpzoo.utilities import add_jitter

class ChromGP(nn.Module):
    def __init__(self, gp, kernel, noise=0.1, jitter=1e-5):
        super().__init__()
        self.gp = gp
        self.kernel = kernel
        self.jitter = jitter
        self.noise = nn.Parameter(torch.tensor(noise))

    def forward(self, X, E=1, verbose=False, idx=None, **kwargs):
        N = len(X)
        noise = torch.nn.functional.softplus(self.noise)  # ensure positive

        if idx is not None and hasattr(self.gp, 'forward_train'):
            qZ, qU, pU = self.gp.forward_train(X, idx=idx, verbose=verbose, **kwargs)
        else:
            qZ, qU, pU = self.gp(X, verbose=verbose, **kwargs)
        F = qZ.rsample((E,)) # (E, D, N)
        F = F.transpose(-2, -1)  # (E, N, D)

        # Process F (can be overridden in subclasses)
        Z = self.process_F(X, F)
        Z = torch.squeeze(Z)

        Kzz = self.kernel.forward(Z, Z)
        Kzz = Kzz.contiguous()
        Kzz = add_jitter(Kzz, self.jitter)
        Kzz.view(-1)[::N+1] += (noise**2)

        pY = distributions.MultivariateNormal(torch.zeros_like(torch.squeeze(X)), Kzz)

        return pY, qZ, qU, pU

    def process_F(self, X, F):
        """
        Process F. Subclasses can override this method to customize behavior.

        Args:
            X: Input tensor
            F: Latent positions tensor

        Returns:
            Processed Z tensor
        """
        Z = F
        return Z


class IntegratedForceGP(ChromGP):
    def __init__(self, gp_force, kernel, noise=0.1, jitter=1e-5):
        super().__init__(gp_force, kernel, noise, jitter)

    def forward(self, X, verbose=False, **kwargs):
        X = X.view(-1, 1)
        return super().forward(X, verbose=verbose, **kwargs)

    def process_F(self, X, F):
        """
        Override process_Z to integrate force F(x) to obtain 3D positions Z(x).

        Args:
            X: (N, 1) tensor of input positions (not necessarily sorted)
            F: (B, N, D) sampled force vectors

        Returns:
            Z: (B, N, D) integrated positions, same order as input X
        """
        B, N, D = F.shape

        # Sort X and reorder F accordingly
        X_sorted, idx = torch.sort(X.view(-1), dim=0)
        F_sorted = F[:, idx]  # (B, N, D)

        # Trapezoidal integration
        dx = X_sorted[1:] - X_sorted[:-1]                   # (N-1,)
        mid = 0.5 * (F_sorted[:, :-1] + F_sorted[:, 1:])    # (B, N-1, D)
        dZ = mid * dx.view(1, -1, 1)                        # (B, N-1, D)

        Z_sorted = torch.cat([torch.zeros((B, 1, D), device=F.device), torch.cumsum(dZ, dim=1)], dim=1)

        # Reorder to match original X
        unsort_idx = torch.argsort(idx)
        Z = Z_sorted[:, unsort_idx]  # (B, N, D)

        return Z
