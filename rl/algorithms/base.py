from abc import ABC, abstractmethod

__all__ = ['Algorithm']


class Algorithm(ABC):
    """Base class for reinforcement learning algorithms."""

    @abstractmethod
    def start(self, *args, **kwargs) -> None:
        """Start the algorithm, i.e. setup variables etc. The arguments of this
        method are algorithm-specific."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Perform an update step of the algorithm. The arguments of this method
        are algorithm-specific."""
        pass
