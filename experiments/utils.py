import collections
import time
from typing import Any, Dict, List, MutableMapping, NamedTuple, Optional, \
    Sequence, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

__all__ = ['set_seed', 'set_current_device', 'flatten_dict', 'add_scalars', 'extract_episode_stats', 'get_env_stats',
           'StatsLogger']

T = TypeVar('T', bound=NamedTuple)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_current_device(device_id: Optional[str]) -> torch.device:
    if device_id is None:
        return torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(device_id)
        if device.type == 'cuda':
            assert torch.cuda.is_available(), 'Device set to CUDA, but CUDA is not available'
            if device.index is None:
                device = torch.device(device_id, torch.cuda.current_device())
            else:
                torch.cuda.set_device(device)
        return device


def flatten_dict(
        d: MutableMapping[str, Any],
        separator: str,
        parent_key: Optional[str] = None) -> Dict[str, Any]:
    items = []
    for (key, value) in d.items():
        key = key if parent_key is None else parent_key + separator + key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_dict(value, separator, parent_key=key).items())
        else:
            items.append((key, value))
    return dict(items)


def add_scalars(
        summary_writer: SummaryWriter,
        scalars: Dict[str, float],
        global_step: int,
        walltime: float = None) -> None:
    walltime = time.time() if walltime is None else walltime
    for (key, value) in scalars.items():
        summary_writer.add_scalar(key, value, global_step, walltime)


def extract_episode_stats(
        info_batch: Sequence[Tuple[Dict[str, Any], ...]]
) -> Tuple[List[float], List[int], Optional[int], Optional[int]]:

    flat_infos = [info_i for info_t in info_batch for info_i in info_t]
    returns = []
    lengths = []
    return_min = None
    return_max = None
    for info in flat_infos:
        if 'episode' in info:
            episode_info = info['episode']
            return_ = episode_info['r']
            length = episode_info['l']
            returns.append(return_)
            lengths.append(length)
            return_min = return_ if return_min is None else min(return_, return_min)
            return_max = return_ if return_max is None else max(return_, return_max)
    return returns, lengths, return_min, return_max


def get_env_stats(
        info: Sequence[Tuple[Dict[str, Any], ...]],
        std: bool = True) -> Dict[str, Tensor]:
    returns, lengths, return_min, return_max = extract_episode_stats(info)
    env_stats = {}
    if len(returns) > 0:
        returns = np.array(returns)
        lengths = np.array(lengths)
        env_stats.update({
            'episode_return_mean': torch.tensor(np.mean(returns)),
            'episode_length_mean': torch.tensor(np.mean(lengths)),
            'episode_return_min': torch.tensor(return_min),
            'episode_return_max': torch.tensor(return_max)
        })
        if std:
            env_stats['episode_return_std'] = torch.tensor(np.std(returns))
    return env_stats


class StatsLogger:

    def __init__(
            self,
            summary_writer: SummaryWriter,
            global_step: int = 0,
            defer_synchronize_steps: int = 0) -> None:
        self.summary_writer = summary_writer
        self.global_step = global_step
        self.defer_synchronize_steps = defer_synchronize_steps
        self._asynchronous_steps = 0
        self._stats_history = []

    def log(self, stats: Dict[str, Any], num_steps: int):
        self.global_step += num_steps
        stats = flatten_dict(stats, '/')
        walltime = time.time()
        self._stats_history.append((stats, self.global_step, walltime))
        self._asynchronous_steps += 1
        if self._asynchronous_steps >= self.defer_synchronize_steps:
            self.synchronize()

    def synchronize(self):
        for (stats, global_step, walltime) in self._stats_history:
            stats = {tag: tensor.item() for (tag, tensor) in stats.items()}
            add_scalars(self.summary_writer, stats, global_step, walltime)
        self._stats_history.clear()
        self._asynchronous_steps = 0
