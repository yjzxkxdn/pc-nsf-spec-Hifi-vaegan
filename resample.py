import itertools
import json
import multiprocessing
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Union

import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils.config_utils import read_full_config
from utils.wav2F0 import PITCH_EXTRACTORS_NAME_TO_ID, get_pitch
from utils.wav2spectrogram import PitchAdjustableSpectrogram
from utils import HParams


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav2spec(config: dict, source: pathlib.Path, save_path: pathlib.Path) -> Tuple[bool, Union[pathlib.Path, str]]:
    spec_transform = PitchAdjustableSpectrogram(
        sample_rate=config['sampling_rate'],
        n_fft=config['fft_size'],
        win_length=config['win_size'],
        hop_length=config['hop_size'],
    )
    try:
        audio, sr = torchaudio.load(str(source))
        pe_name = config['pe']
        pe_id = PITCH_EXTRACTORS_NAME_TO_ID[pe_name]
        if sr > config['sampling_rate']:
            audio = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=config['sampling_rate'],
                lowpass_filter_width=128)(audio)
        elif sr < config['sampling_rate']:
            return False, f"Error: sample rate mismatching in \'{source}\' ({sr} != {config['sampling_rate']})."
        spec = dynamic_range_compression_torch(spec_transform(audio))
        f0, uv = get_pitch(pe_name, audio.numpy()[0], length=len(spec[0].T), hparams=config, interp_uv=True)
        if f0 is None:
            return False, f"Error: failed to get pitch from \'{source}\'."
        np.savez(save_path, audio=audio[0].numpy(), spec=spec[0].T, f0=f0, uv=uv, pe=pe_id)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return False, f"Error: {e.__class__.__name__}: {e}"
    return True, save_path


@click.command(help='')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
@click.option('--num_cpu', required=False, metavar='DIR2', help='Number of CPU cores to use')
@click.option('--strx', required=False, metavar='DIR4', help='Whether to use strict path')  # 1 代表开   0代表关
def runx(config, num_cpu, strx):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    config = pathlib.Path(config)
    config = json.loads(config.read_text())
    config = HParams(**config)
    print(config)
    # print_config(config)
    if strx is None:
        strx = 1
    else:
        strx = int(strx)
    if strx == 1:
        strx = True
    else:
        strx = False

    val_data_filename_set = set()
    val_outlist = preprocess(config=config, input_path=config.validation_files, output_path=pathlib.Path(config.model_dir) / 'valid', num_cpu=num_cpu, st_path=strx)
    val_data_filename_set.update(val_outlist)

    train_data_filename_set = set()
    train_outlist = preprocess(config=config, input_path=config.training_files, output_path=pathlib.Path(config.model_dir) / 'train', num_cpu=num_cpu, st_path=strx)
    train_data_filename_set.update(train_outlist)

    outp = pathlib.Path(config['DataIndexPath'])
    assert not outp.exists() or outp.is_dir(), f'Path \'{outp}\' is not a directory.'
    outp.mkdir(parents=True, exist_ok=True)
    train_name = 'train'
    val_name = 'valid'
    with open(outp / train_name, 'w', encoding='utf8') as f:
        [print(p, file=f) for p in sorted(train_data_filename_set)]
    with open(outp / val_name, 'w', encoding='utf8') as f:
        [print(p, file=f) for p in sorted(val_data_filename_set)]


def preprocess(config, input_path, output_path, num_cpu, st_path):
    if st_path:
        input_path = pathlib.Path(input_path).resolve()
        output_path = pathlib.Path(output_path).resolve()
    else:
        input_path = pathlib.Path(input_path)
        output_path = pathlib.Path(output_path)

    assert not output_path.exists() or output_path.is_dir(), f'Path \'{output_path}\' is not a directory.'
    output_path.mkdir(parents=True, exist_ok=True)

    if num_cpu is None:
        num_cpu = 5
    else:
        num_cpu = int(num_cpu)

    args = []
    for wav_file in tqdm(
            itertools.chain(input_path.rglob('*.wav'), input_path.rglob('*.flac')),
            desc="Enumerating files", leave=False
    ):
        save_path = output_path / wav_file.relative_to(input_path).with_suffix('.npz')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        args.append((
            config,
            wav_file,
            save_path,
        ))
    

    filenames = []
    completed = 0
    failed = 0
    try:
        with ProcessPoolExecutor(max_workers=num_cpu) as executor:
            tasks = [
                executor.submit(wav2spec, *a)
                for a in tqdm(args, desc="Submitting tasks", leave=False)
            ]
            with tqdm(as_completed(tasks), desc="Preprocessing", total=len(tasks)) as progress:
                for task in progress:
                    succeeded, result = task.result()
                    if succeeded:
                        result: pathlib.Path
                        filenames.append(result.as_posix())
                        completed += 1
                    else:
                        result: str
                        progress.write(result)
                        failed += 1
                    progress.set_description(
                        "Preprocessing ({} completed, {} failed)".format(completed, failed)
                    )
    except KeyboardInterrupt:
        exit(-1)

    return filenames


if __name__ == '__main__':
    runx()
