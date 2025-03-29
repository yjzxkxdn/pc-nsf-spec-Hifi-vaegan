from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pathlib
import re
import sys
import types
from collections import OrderedDict
from typing import Any, Dict, TypeVar, Union

import numpy as np
import torch

from utils.training_utils import get_latest_checkpoint_path


def tensors_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if type(v) is dict:
            v = tensors_to_scalars(v)
        new_metrics[k] = v
    return new_metrics


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = ((max(v.size(0) for v in values) if max_len is None else max_len), *values[0].shape[1:])
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        res[i, :len(v), ...] = v
    return res


def random_continuous_masks(*shape: int, dim: int, device: str | torch.device = 'cpu'):
    start, end = torch.sort(
        torch.randint(
            low=0, high=shape[dim] + 1, size=(*shape[:dim], 2, *((1,) * (len(shape) - dim - 1))), device=device
        ).expand(*((-1,) * (dim + 1)), *shape[dim + 1:]), dim=dim
    )[0].split(1, dim=dim)
    idx = torch.arange(
        0, shape[dim], dtype=torch.long, device=device
    ).reshape(*((1,) * dim), shape[dim], *((1,) * (len(shape) - dim - 1)))
    masks = (idx >= start) & (idx < end)
    return masks


def _is_batch_full(batch, num_frames, max_batch_frames, max_batch_size):
    if len(batch) == 0:
        return 0
    if len(batch) == max_batch_size:
        return 1
    if num_frames > max_batch_frames:
        return 1
    return 0


def batch_by_size(
        indices, num_frames_fn, max_batch_frames=80000, max_batch_size=48,
        required_batch_size_multiple=1
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_frames_fn (callable): function that returns the number of frames at
            a given index
        max_batch_frames (int, optional): max number of frames in each batch
            (default: 80000).
        max_batch_size (int, optional): max number of sentences in each
            batch (default: 48).
        required_batch_size_multiple: require the batch size to be multiple
            of a given number
    """
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_frames = num_frames_fn(idx)
        sample_lens.append(num_frames)
        sample_len = max(sample_len, num_frames)
        assert sample_len <= max_batch_frames, (
            "sentence at index {} of size {} exceeds max_batch_samples "
            "limit of {}!".format(idx, sample_len, max_batch_frames)
        )
        num_frames = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_frames, max_batch_frames, max_batch_size):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        # the signature contains definitions like **kwargs, so there is no need to filter
        return dict_to_filter.copy()
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict


def load_ckpt(
        cur_model, ckpt_base_dir, ckpt_steps=None,
        prefix_in_ckpt='model', key_in_ckpt='state_dict',
        strict=True, device='cpu'
):
    if not isinstance(ckpt_base_dir, pathlib.Path):
        ckpt_base_dir = pathlib.Path(ckpt_base_dir)
    if ckpt_base_dir.is_file():
        checkpoint_path = [ckpt_base_dir]
    elif ckpt_steps is not None:
        checkpoint_path = [ckpt_base_dir / f'model_ckpt_steps_{int(ckpt_steps)}.ckpt']
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(
            [
                ckpt_file
                for ckpt_file in base_dir.iterdir()
                if ckpt_file.is_file() and re.fullmatch(r'model_ckpt_steps_\d+\.ckpt', ckpt_file.name)
            ],
            key=lambda x: int(re.search(r'\d+', x.name).group(0))
        )
    assert len(checkpoint_path) > 0, f'| ckpt not found in {ckpt_base_dir}.'
    checkpoint_path = checkpoint_path[-1]
    ckpt_loaded = torch.load(checkpoint_path, map_location=device)
    if key_in_ckpt is None:
        state_dict = ckpt_loaded
    else:
        state_dict = ckpt_loaded[key_in_ckpt]
    if prefix_in_ckpt is not None:
        state_dict = OrderedDict({
            k[len(prefix_in_ckpt) + 1:]: v
            for k, v in state_dict.items() if k.startswith(f'{prefix_in_ckpt}.')
        })
    if not strict:
        cur_model_state_dict = cur_model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    cur_model.load_state_dict(state_dict, strict=strict)
    shown_model_name = 'state dict'
    if prefix_in_ckpt is not None:
        shown_model_name = f'\'{prefix_in_ckpt}\''
    elif key_in_ckpt is not None:
        shown_model_name = f'\'{key_in_ckpt}\''
    print(f'| load {shown_model_name} from \'{checkpoint_path}\'.')


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    # num_params(model, model_name=model_name)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters


def build_object_from_class_name(cls_str, parent_cls, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'

    return cls_type(*args, **filter_kwargs(kwargs, cls_type))


def build_lr_scheduler_from_config(optimizer, scheduler_args):
    try:
        # PyTorch 2.0+
        from torch.optim.lr_scheduler import LRScheduler as LRScheduler
    except ImportError:
        # PyTorch 1.X
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    def helper(params):
        if isinstance(params, list):
            return [helper(s) for s in params]
        elif isinstance(params, dict):
            resolved = {k: helper(v) for k, v in params.items()}
            if 'cls' in resolved:
                if (
                    resolved["cls"] == "torch.optim.lr_scheduler.ChainedScheduler"
                    and scheduler_args["scheduler_cls"] == "torch.optim.lr_scheduler.SequentialLR"
                ):
                    raise ValueError(f"ChainedScheduler cannot be part of a SequentialLR.")
                resolved['optimizer'] = optimizer
                obj = build_object_from_class_name(
                    resolved['cls'],
                    LRScheduler,
                    **resolved
                )
                return obj
            return resolved
        else:
            return params

    resolved = helper(scheduler_args)
    resolved['optimizer'] = optimizer
    return build_object_from_class_name(
        scheduler_args['scheduler_cls'],
        LRScheduler,
        **resolved
    )


def simulate_lr_scheduler(optimizer_args, scheduler_args, step_count, num_param_groups=1):
    optimizer = build_object_from_class_name(
        optimizer_args['optimizer_cls'],
        torch.optim.Optimizer,
        [{'params': torch.nn.Parameter(), 'initial_lr': optimizer_args['lr']} for _ in range(num_param_groups)],
        **optimizer_args
    )
    scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
    scheduler.optimizer._step_count = 1
    for _ in range(step_count):
        scheduler.step()
    return scheduler.state_dict()


def remove_suffix(string: str, suffix: str):
    #  Just for Python 3.8 compatibility, since `str.removesuffix()` API of is available since Python 3.9
    if string.endswith(suffix):
        string = string[:-len(suffix)]
    return string

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logger = logging

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("Load checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)

def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))
    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))
    sort_key = time_key if sort_by_time else name_key
    def x_sorted(_x):
        return sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")], key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
                (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")
    def del_routine(x):
        return [os.remove(x), del_info(x)]
    [del_routine(fn) for fn in to_del]

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10,2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                    interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def load_wav_to_torch(full_path):
    sampling_rate, data = os.read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def get_hparams(config_path, init=True):
    with open(config_path, "r") as f:
        data = f.read()
        config = json.loads(data)
        if init:
            config_save_path = os.path.join(config["data"]["model_dir"], "config.json")
            with open(config_save_path, "w") as f:
                f.write(data)
    hparams = HParams(**config)
    return hparams

def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams =HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_file(config_path, infer_mode = False):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams =HParams(**config) if not infer_mode else InferHParams(**config)
    return hparams

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger
class HParams:
    def __init__(self, **kwargs):
        self._flatten_and_set(kwargs)

    def _flatten_and_set(self, d):
        """
        递归地将嵌套字典扁平化，并设置为对象属性。
        :param d: 输入的字典
        """
        for k, v in d.items():
            if isinstance(v, dict):  
                self._flatten_and_set(v)
            else:
                if hasattr(self, k):
                    raise ValueError(f"Duplicate key found: {k}. Please ensure all keys are unique.")
                setattr(self, k, v)   

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self,index):
        return self.__dict__.get(index)

class InferHParams(HParams):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = InferHParams(**v)
      self[k] = v

  def __getattr__(self,index):
    return self.get(index)
  