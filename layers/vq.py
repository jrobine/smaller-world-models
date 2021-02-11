from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from layers.ema import ExponentialMovingAverages

__all__ = ['VectorQuantizer']


class VectorQuantizer(nn.Module):
    """Quantizes vectors using a list of embedding vectors, as used for the vector quantized-variational autoencoder
    (VQ-VAE) [1]. Implementation adopted from TensorFlow version by DeepMind [2].

    Arguments:
        num_embeddings (int): The number of embedding vectors.
        embedding_size (int): The size of each embedding vector.
        commitment_cost (float, optional): The commitment cost used in the loss. Defaults to 0.25.
        exponential_moving_averages (bool, optional): Whether or not to use exponential moving averages to update the
            embedding vectors. For more details, see Appendix A.1 in [1]. Defaults to ``False``.
        ema_decay (float, optional): The decay parameter used for the exponential moving averages, if used.
            Defaults to 0.99.
        ema_epsilon (float, optional): The epsilon parameter used for the exponential moving averages, if used.
            Defaults to 1e-5.

    Inputs: z_e
        **z_e** of shape `(d1, ..., dn, embedding_size)`: Tensor containing the vectors which will be quantized.

    Outputs: z, z_q
        **z** of shape `(d1, ..., dn)`: Tensor containing the indices of the nearest embedding vectors.
        **z_q** of shape `(d1, ..., dn, embedding_size)`: Tensor containing the quantized vectors.

    References:
        [1] Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu (2018). Neural Discrete Representation Learning.
            arXiv preprint: https://arxiv.org/abs/1711.00937
        [2] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            commitment_cost: float = 0.25,
            exponential_moving_averages: bool = False,
            ema_decay: Optional[float] = 0.99,
            ema_epsilon: Optional[float] = 1e-5) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(num_embeddings, embedding_size))
        limit = np.sqrt(3. / num_embeddings)  # LeCun's uniform initialization
        nn.init.uniform_(self.embeddings, -limit, limit)

        self.register_buffer('_commitment_cost', torch.tensor(commitment_cost, dtype=torch.float))

        self._exponential_moving_averages = exponential_moving_averages
        if exponential_moving_averages:
            assert ema_decay is not None and ema_epsilon is not None
            self._ema_epsilon = ema_epsilon

            self._ema_dw = ExponentialMovingAverages(
                torch.zeros(num_embeddings, embedding_size),
                decay=ema_decay, bias_correction=True)

            self._ema_cluster_sizes = ExponentialMovingAverages(
                torch.zeros(num_embeddings),
                decay=ema_decay, bias_correction=False)

            self.embeddings.requires_grad_(False)

            # nn.functional.one_hot seems to be slow (at least PyTorch 1.6.0),
            # so we cache the results and index the variable
            self.register_buffer(
                '_one_hot_cache', nn.functional.one_hot(torch.arange(num_embeddings), num_embeddings).float(),
                persistent=False)

    @property
    def num_embeddings(self) -> int:
        """Returns the number of embedding vectors."""
        return self.embeddings.shape[0]

    @property
    def embedding_size(self) -> int:
        """Returns the size of each embedding vector."""
        return self.embeddings.shape[1]

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.nearest_indices(z_e)
        z_q = self.lookup(z)
        return z, z_q

    def nearest_indices(self, z_e: Tensor) -> Tensor:
        """TODO docstring"""
        assert z_e.shape[-1] == self.embedding_size
        w = self.embeddings
        z_e_flat = z_e.reshape(-1, self.embedding_size)
        distances_flat = z_e_flat.square().sum(1).unsqueeze(1) - 2 * (z_e_flat @ w.T) + w.square().sum(1).unsqueeze(0)
        z = torch.argmin(distances_flat, dim=-1).reshape(z_e.shape[:-1])
        return z

    def lookup(self, z: Tensor) -> Tensor:
        """TODO docstring"""
        z_q = nn.functional.embedding(z, self.embeddings)
        return z_q

    def compute_loss(
            self,
            z_e: Tensor,
            z: Tensor,
            z_q: Tensor,
            update_ema: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        """TODO docstring"""
        z_e_loss = nn.functional.mse_loss(z_q.detach(), z_e)
        if self._exponential_moving_averages:
            loss = self._commitment_cost * z_e_loss
            stats = {
                'vq_loss': loss.detach().clone(),
                'z_e_loss': z_e_loss.detach().clone()
            }
            if update_ema:
                self.update_ema(z_e, z)
        else:
            z_q_loss = nn.functional.mse_loss(z_q, z_e.detach())
            loss = z_q_loss + self._commitment_cost * z_e_loss
            stats = {
                'vq_loss': loss.detach().clone(),
                'z_e_loss': z_q_loss.detach().clone(),
                'z_q_loss': z_q_loss.detach().clone()
            }
        return loss, stats

    def update_ema(self, z_e: Tensor, z: Tensor) -> None:
        """TODO docstring"""
        assert self._exponential_moving_averages
        with torch.no_grad():
            flat_z_e = z_e.reshape(-1, self.embedding_size)
            flat_z = z.reshape(-1)
            flat_one_hot_z = self._one_hot_cache[flat_z]
            # without cache:
            # flat_one_hot_z = nn.functional.one_hot(
            #     flat_z, num_classes=num_embeddings)

            cluster_sizes = flat_one_hot_z.sum(0)
            # sum of closest input vectors
            dw = flat_one_hot_z.T @ flat_z_e
            average_cluster_sizes = self._ema_cluster_sizes.update(cluster_sizes)
            average_dw = self._ema_dw.update(dw)

            n = average_cluster_sizes.sum()
            stable_average_cluster_sizes = \
                (average_cluster_sizes + self._ema_epsilon) / (n + self.num_embeddings * self._ema_epsilon) * n

            self.embeddings.data = average_dw / stable_average_cluster_sizes.unsqueeze(1)
