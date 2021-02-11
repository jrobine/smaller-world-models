import random
from typing import Generator, KeysView, NamedTuple, Sequence, Type, TypeVar, Optional

import torch
from torch import Tensor

__all__ = ['TrajectoryBuffer']

T = TypeVar('T', bound=NamedTuple)


class TrajectoryBuffer:
    """Utility class for storing and sampling from multiple sequences of
    experience, e.g., entire episodes.

    Arguments:
        fields (sequence of str): The field names of the named tuple values that
            will be saved in this buffer. For more information on named tuples,
            see :method:`rl.utils.EnvSampler.sample()`.
        device (torch.device, optional): TODO
    """

    def __init__(self, fields: Sequence[str], device: Optional[torch.device] = None) -> None:
        self._trajectories = {field: [] for field in fields}
        self._lengths = []
        self._device = device

    @property
    def fields(self) -> KeysView[str]:
        """Returns the field names that will be saved in this buffer."""
        return self._trajectories.keys()

    @property
    def device(self) -> Optional[torch.device]:
        """TODO"""
        return self._device

    @property
    def num_trajectories(self) -> int:
        """Returns the number of trajectories that are currently stored in this
        buffer."""
        return len(self._lengths)

    def add_trajectory(self, trajectory: NamedTuple) -> None:
        """Adds a trajectory (e.g. an episode) to this buffer.

        Arguments:
            trajectory (NamedTuple): The named tuple containing the tensors of
                shape `(seq_len, 1, *)`.
        """
        ref_field = trajectory._fields[0]
        ref_value = getattr(trajectory, ref_field)
        length = ref_value.shape[0]
        self._lengths.append(length)

        for (field, trajectory_list) in self._trajectories.items():
            value = getattr(trajectory, field)
            assert isinstance(value, Tensor), 'All values must be tensors.'
            assert value.shape[1] == 1, 'Batch size must be 1.'
            assert value.shape[0] == length, \
                'Length must be the same for all values.'
            value = value.squeeze(1)
            if self._device is not None and value.device != self._device:
                value = value.to(self._device)
            trajectory_list.append(value)

    def generate_minibatches(
            self,
            named_tuple_cls: Type[T],
            minibatch_size: int,
            minibatch_steps: int,
            max_random_offset: int,
            shuffle: bool,
            drop_last_batch: bool,
            drop_last_time: bool,
            last_trajectory_first: bool = False,
            device: Optional[torch.device] = None) -> Generator[T, None, None]:
        """TODO docstring"""
        fitting_indices = [i for i in range(self.num_trajectories)
                           if self._lengths[i] >= minibatch_steps]

        index_tuples = []  # tuples of (trajectory index, time index)
        latest_index_tuples = []  # separate tuples of the latest trajectory
        for trajectory_index in fitting_indices:
            length = self._lengths[trajectory_index]
            random_offset = random.randint(
                0, min(max_random_offset, max(length - minibatch_steps - 1, 0)))
            for time_index in range(random_offset, length, minibatch_steps):
                if drop_last_time and length - time_index < minibatch_steps:
                    continue
                if last_trajectory_first and \
                        trajectory_index == self.num_trajectories - 1:
                    latest_index_tuples.append((trajectory_index, time_index))
                else:
                    index_tuples.append((trajectory_index, time_index))

        # TODO check how many are dropped

        if shuffle:
            random.shuffle(index_tuples)
            random.shuffle(latest_index_tuples)

        # insert index tuples of the latest trajectory at the front to ensure
        # that the latest trajectory is iterated over, even if not all
        # minibatches are used
        for index_tuple in reversed(latest_index_tuples):
            index_tuples.insert(0, index_tuple)

        if drop_last_batch:
            for i in range(len(index_tuples) % minibatch_size):
                index_tuples.pop(-1)

        fields = named_tuple_cls._fields
        assert all(field in self._trajectories for field in fields)

        while len(index_tuples) > 0:
            minibatch_index_tuples = []
            for i in range(min(minibatch_size, len(index_tuples))):
                minibatch_index_tuples.append(index_tuples.pop(0))

            minibatch_values = {field: [] for field in fields}
            for (trajectory_index, time_index) in minibatch_index_tuples:
                for field in fields:
                    value = self._trajectories[field][trajectory_index]
                    minibatch_values[field].append(
                        value[time_index:time_index + minibatch_steps])

            minibatch_values = {field: torch.stack(values, 1)
                                for field, values in minibatch_values.items()}
            if device is not None:
                minibatch_values = {field: tensor.to(device) if tensor.device != device else tensor
                                    for field, tensor in minibatch_values.items()}
            yield named_tuple_cls(**minibatch_values)
