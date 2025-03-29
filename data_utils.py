import pathlib
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

from utils.wav2F0 import PITCH_EXTRACTORS_ID_TO_NAME, get_pitch
from utils.wav2spectrogram import PitchAdjustableSpectrogram

def get_max_f0_from_config(config: dict):
    if config.mini_nsf:
        source_sr = config.sampling_rate / int(np.prod(config.upsample_rates[2:]))
    else:
        source_sr = config.sampling_rate
    max_f0 = source_sr / 2
    return max_f0

def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9), dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav_aug(wav, hop_size, speed=1):
    orig_freq = int(np.round(hop_size * speed))
    new_freq = hop_size
    resample = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=128
    )
    wav_resampled = resample(wav)
    del resample
    return wav_resampled

class nsf_HiFigan_dataset(Dataset):

    def __init__(self, config ,data_dir,infer=False):
    #def __init__(self, config ,data_dir,world_siz,rank,infer=False):
        super().__init__()
        self.config = config

        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        with open(self.data_dir, 'r', encoding='utf8') as f:
            fills = f.read().strip().split('\n')
        self.data_index = fills
        self.infer = infer
        self.volume_aug = self.config.volume_aug
        self.volume_aug_prob = self.config.volume_aug_prob if not infer else 0
        self.key_aug = self.config.key_aug
        self.key_aug_prob = self.config.key_aug_prob
        if self.key_aug:
            self.spec_transform = PitchAdjustableSpectrogram(
                sample_rate=config.sampling_rate,
                n_fft=config.fft_size,
                win_length=config.win_size,
                hop_length=config.hop_size,
            )
        self.max_f0 = get_max_f0_from_config(config)

    def __getitem__(self, index):
        sample = self.get_data(index)
        if sample['f0'].max() >= self.max_f0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return sample

    def __len__(self):
        return len(self.data_index)

    def get_data(self, index):
        data_path = pathlib.Path(self.data_index[index])
        data = np.load(data_path)
        pe_name = PITCH_EXTRACTORS_ID_TO_NAME[int(data['pe'])]
        if self.infer or not self.key_aug or random.random() > self.key_aug_prob:
            return {'f0': data['f0'], 'spec': data['spec'], 'audio': data['audio']}
        else:
            speed = random.uniform(self.config['aug_min'], self.config['aug_max'])
            crop_spec_frames = int(np.ceil((self.config['crop_spec_frames'] + 4) * speed))
            samples_per_frame = self.config['hop_size']
            crop_wav_samples = crop_spec_frames * samples_per_frame
            if crop_wav_samples >= data['audio'].shape[0]:
                return {'f0': data['f0'], 'spec': data['spec'], 'audio': data['audio']}
            start = random.randint(0, data['audio'].shape[0] - 1 - crop_wav_samples)
            end = start + crop_wav_samples
            audio = data['audio'][start:end]
            audio_aug = wav_aug(torch.from_numpy(audio), self.config["hop_size"], speed=speed)
            mel_aug = dynamic_range_compression_torch(self.spec_transform(audio_aug[None, :]))
            f0, _ = get_pitch(
                pe_name, audio, length=mel_aug.shape[-1], hparams=self.config,
                speed=speed, interp_uv=True
            )
            if f0 is None:
                return {'f0': data['f0'], 'spec': data['spec'], 'audio': data['audio']}
            audio_aug = audio_aug[2 * samples_per_frame: -2 * samples_per_frame].numpy()
            mel_aug = mel_aug[0, :, 2:-2].T.numpy()
            f0_aug = f0[2:-2] * speed
            return {'f0': f0_aug, 'spec': mel_aug, 'audio': audio_aug}


class TextAudioCollate:
    def __init__(self, config, infer=False):
        self.config = config
        self.infer = infer
    def __call__(self, minibatch):
        samples_per_frame = self.config['hop_size']
        if self.infer:
            crop_spec_frames = 0
        else:
            crop_spec_frames = self.config['crop_spec_frames']

        for record in minibatch:

            # Filter out records that aren't long enough.
            if record['spec'].shape[0] < crop_spec_frames:
                del record['spec']
                del record['audio']
                del record['f0']
                continue
            elif record['spec'].shape[0] == crop_spec_frames:
                start = 0
            else:
                start = random.randint(0, record['spec'].shape[0] - 1 - crop_spec_frames)
            end = start + crop_spec_frames
            if self.infer:
                record['spec'] = record['spec'].T
                record['f0'] = record['f0']
            else:
                record['spec'] = record['spec'][start:end].T
                record['f0'] = record['f0'][start:end]
            start *= samples_per_frame
            end *= samples_per_frame
            if self.infer:
                cty = (len(record['spec'].T) * samples_per_frame)
                record['audio'] = record['audio'][:cty]
                record['audio'] = np.pad(record['audio'], (
                    0, (len(record['spec'].T) * samples_per_frame) - len(record['audio'])),
                                         mode='constant')
                pass
            else:
                # record['spec'] = record['spec'][start:end].T
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])),
                                         mode='constant')

        if self.config.volume_aug:
            for record in minibatch:
                if record.get('audio') is None:
                    # del record['spec']
                    # del record['audio']
                    # del record['pemel']
                    # del record['uv']
                    continue
                audio = record['audio']
                audio_mel = record['spec']

                if random.random() < self.config.volume_aug_prob:
                    max_amp = float(np.max(np.abs(audio))) + 1e-5
                    max_shift = min(3, np.log(1 / max_amp))
                    log_mel_shift = random.uniform(-3, max_shift)
                    # audio *= (10 ** log_mel_shift)
                    audio *= np.exp(log_mel_shift)
                    audio_mel += log_mel_shift

                audio_mel = torch.clamp(torch.from_numpy(audio_mel), min=np.log(1e-5)).numpy()
                record['audio'] = audio
                record['spec'] = audio_mel

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])

        spectrogram = np.stack([record['spec'] for record in minibatch if 'spec' in record])
        f0 = np.stack([record['f0'] for record in minibatch if 'f0' in record])

        return {
            'audio': torch.from_numpy(audio).unsqueeze(1),
            'spec': torch.from_numpy(spectrogram), 'f0': torch.from_numpy(f0),
        }