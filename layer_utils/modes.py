from typing import Callable, Type, TypeVar, Union

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical, ContinuousBernoulli, Distribution, Independent, Normal

__all__ = ['register_mode', 'has_mode', 'mode']

_MODE_REGISTRY = dict()

T = TypeVar('T', bound=Distribution)


def register_mode(type_p: Type[T]) -> Callable[[Callable[[T], Tensor]], Callable[[T], Tensor]]:
    """Decorator to register a function that computes the mode for the given type of distribution. The registered
    function can be called via `mode()`.

    Arguments:
        type_p (type): Type of the torch.distributions.Distribution for which the mode is registered.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(f'Expected type_p to be a Distribution subclass '
                        f'but got {type_p}')

    def decorator(fun: Callable[[T], Tensor]) -> Callable[[T], Tensor]:
        _MODE_REGISTRY[type_p] = fun
        return fun

    return decorator


def has_mode(p_type: Union[Type[Distribution], Distribution]) -> bool:
    """Returns whether a mode for the specified type of distribution has been registered before.

    Arguments:
        p_type (type or Distribution): The type of distribution to check.
    """
    if isinstance(p_type, Distribution):
        p_type = type(p_type)
    return p_type in _MODE_REGISTRY


def mode(p: Distribution) -> Tensor:
    """Computes the mode of the given probability distribution. The function has to be registered before via
    `register_mode()`. Use `has_mode()` to check whether a mode has been registered.

    Arguments:
        p (Distribution): The distribution object of which the mode is computed.

    Returns:
        Tensor containing the mode of the distribution.

    Raises:
        NotImplementedError, if no mode has been registered for this type of distribution.
    """
    try:
        fun = _MODE_REGISTRY[type(p)]
    except KeyError:
        raise NotImplementedError()
    if fun is NotImplemented:
        raise NotImplementedError()
    return fun(p)


@register_mode(Independent)
def _mode_independent(p: Independent) -> Tensor:
    return mode(p.base_dist)


@register_mode(Bernoulli)
def _mode_bernoulli(p: Bernoulli) -> Tensor:
    return torch.round(p.probs).long()


@register_mode(Categorical)
def _mode_categorical(p: Categorical) -> Tensor:
    return torch.argmax(p.logits, dim=-1)


@register_mode(ContinuousBernoulli)
def _mode_continuous_bernoulli(p: ContinuousBernoulli) -> Tensor:
    return p.mean  # TODO?


@register_mode(Normal)
def _mode_normal(p: Normal) -> Tensor:
    return p.mean
