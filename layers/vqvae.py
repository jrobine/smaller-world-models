import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution, Independent

from layer_utils.modes import mode
from layers.vq import VectorQuantizer

__all__ = ['VQVAE']


class VQVAE(nn.Module, ABC):
    """Base class for VQ-VAEs [1]. Subclasses have to implement `encode()` and `decode()`.

    Arguments:
        num_embeddings (int): The number of embedding vectors.
        embedding_size (int): The size of each embedding vector.
        latent_height (int): The height of the latent variable returned by `encode()`.
        latent_width (int): The width of the latent variable returned by `encode()`.
        exponential_moving_averages (bool, optional): Whether or not to use exponential moving averages to update the
            embedding vectors. For more details, see Appendix A.1 in [1]. Defaults to ``False``.
        commitment_cost (float, optional): The commitment cost used in the loss. Defaults to 0.25.
        ema_decay (float, optional): The decay parameter used for the exponential moving averages, if used.
            Defaults to 0.99.
        ema_epsilon (float, optional): The epsilon parameter used for the exponential moving averages, if used.
            Defaults to 1e-5.

    Inputs: TODO docstring

    Outputs: TODO docstring

    References:
        [1] Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu (2018). Neural Discrete Representation Learning.
            arXiv preprint: https://arxiv.org/abs/1711.00937
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            latent_height: int,
            latent_width: int,
            commitment_cost: float = 0.25,
            exponential_moving_averages: bool = False,
            ema_decay: Optional[float] = 0.99,
            ema_epsilon: Optional[float] = 1e-5) -> None:
        super().__init__()
        self.vq = VectorQuantizer(
            num_embeddings, embedding_size, commitment_cost, exponential_moving_averages, ema_decay, ema_epsilon)
        self.latent_height = latent_height
        self.latent_width = latent_width

        prob = 1. / self.num_embeddings
        kl = -math.log(prob) * (latent_height * latent_width)
        self.register_buffer('_kl', torch.tensor(kl), persistent=False)

    @property
    def num_embeddings(self) -> int:
        """Returns the number of embedding vectors."""
        return self.vq.num_embeddings

    @property
    def embedding_size(self) -> int:
        """Returns the size of each embedding vector."""
        return self.vq.embedding_size

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """TODO docstring"""
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Distribution:
        """TODO docstring"""
        pass

    def prior(self, device: Optional[torch.device] = None) -> Distribution:
        """TODO docstring"""
        if device is None:
            device = next(self.parameters()).device

        h, w = self.latent_shape
        prob = 1 / self.num_embeddings
        probs = torch.full((1, h, w, self.num_embeddings,), prob, dtype=torch.float, device=device)

        return Independent(Categorical(probs=probs), reinterpreted_batch_ndims=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Distribution]:
        z_e = self.encode(x)
        z, z_q = self.quantize(z_e)
        z_q = self.straight_through_estimator(z_q, z_e)
        x_posterior = self.decode(z_q)
        return z_e, z, z_q, x_posterior

    def quantize(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        """TODO docstring"""
        z, z_q = self.vq(z_e)
        return z, z_q

    def straight_through_estimator(self, z_q: Tensor, z_e: Tensor) -> Tensor:
        """TODO docstring"""
        return z_e + (z_q - z_e).detach()

    def encode_and_quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """TODO docstring"""
        z_e = self.encode(x)
        z, z_q = self.vq(z_e)
        return z, z_q

    def lookup(self, z: Tensor) -> Tensor:
        """TODO docstring"""
        z_q = self.vq.lookup(z)
        return z_q

    def compute_loss(
            self,
            x: Tensor,
            z_e: Tensor,
            z: Tensor,
            z_q: Tensor,
            x_posterior: Distribution) -> Tuple[Tensor, Dict[str, Tensor]]:
        """TODO docstring"""
        # ELBO = E[log p(x|z)] - KL(q(z)||p(z))
        log_likelihood = x_posterior.log_prob(x).mean(0)
        elbo = log_likelihood - self._kl

        stats = {'elbo': elbo.detach().clone(),
                 'log_likelihood': log_likelihood.detach().clone(),
                 'kl': self._kl.clone()}

        vq_loss, vq_stats = self.vq.compute_loss(z_e, z, z_q)
        loss = -elbo + vq_loss
        stats.update(vq_stats)

        return loss, stats

    def sample(
            self,
            sample_shape: torch.Size = torch.Size(),
            sample_decoder: bool = False,
            device: Optional[torch.device] = None) -> Tensor:
        """TODO docstring"""
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device
            z_prior = self.prior(device=device)
            z = z_prior.sample(sample_shape)
            x_posterior = self.decode(z)
            x = x_posterior.sample() if sample_decoder else mode(x_posterior)
            return x

    def reconstruct(self, x: Tensor, sample: bool = False) -> Tensor:
        """TODO docstring"""
        with torch.no_grad():
            z_e, z, z_q, x_posterior = self.forward(x)
            x_recon = x_posterior.sample() if sample else mode(x_posterior)
            return x_recon
