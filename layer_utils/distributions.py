import torch
from torch import Tensor
from torch.distributions import Distribution, ContinuousBernoulli, Independent, Normal

__all__ = ['independent_continuous_bernoullis',
           'independent_unit_variance_gaussians']


def independent_continuous_bernoullis(logits: Tensor) -> Distribution:
    """Creates a joint distribution for independent continuous Bernoulli random variables.

    Arguments:
        logits (Tensor): The logits for each of the continuous Bernoulli distributions.

    Returns:
        A torch.distributions.Distribution object that represents the joint distribution.
    """
    return Independent(ContinuousBernoulli(logits=logits), reinterpreted_batch_ndims=logits.ndim - 1)


def independent_unit_variance_gaussians(means: Tensor) -> Distribution:
    """Creates a joint distribution for independent Gaussian variables with unit variance.

    Arguments:
        means (Tensor): The means for each of the Gaussian distributions.

    Returns:
        A torch.distributions.Distribution object that represents the joint distribution.
    """
    stds = torch.ones_like(means)
    return Independent(Normal(means, stds), reinterpreted_batch_ndims=means.ndim - 1)
