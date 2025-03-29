import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

class PitchAdjustableSpectrogram:
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.center = center
        self.hann_window = {}

    def __call__(self, y, key_shift=0, speed=1.0):
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * speed))
        hann_window_key = f"{key_shift}_{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                win_size_new, device=y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((win_size_new - hop_length) // 2),
                int((win_size_new - hop_length+1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length,
            win_length=win_size_new,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * self.win_size / win_size_new

        return spec

    def dynamic_range_compression_torch(self,x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

if __name__=='__main__':
    import glob
    import torchaudio
    from tqdm import tqdm
    # from concurrent.futures import ProcessPoolExecutor
    # import random

    # import re
    # from torch.multiprocessing import Manager, Process, current_process, get_context
    #
    # is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))


    lll = glob.glob(r'D:\propj\Disa\data\opencpop\raw\wavs/**.wav')
    torch.set_num_threads(1)

    for i in tqdm(lll):
        audio, sr = torchaudio.load(i)
        audio = torch.clamp(audio[0], -1.0, 1.0)

        spec_transform=PitchAdjustableSpectrogram()
        with torch.no_grad():
            spectrogram = spec_transform(audio.unsqueeze(0).cuda())*0.434294
            # spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20  #ds 是log10
            # spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))