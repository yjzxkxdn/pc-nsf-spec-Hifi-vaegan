


import numpy as np
import torch
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
from vector_quantize_pytorch import VectorQuantize

import modules.modules as modules
from modules.commons import get_padding, init_weights
from modules.msstftd import MultiScaleSTFTDiscriminator
from utils.wav2spectrogram import PitchAdjustableSpectrogram
from utils import HParams

LRELU_SLOPE = 0.1
class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            print(f"conv1.weight.shape: {c1.weight.shape}, conv2.weight.shape: {c2.weight.shape}")
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            print(f"xt_conv1.shape: {xt.shape}")
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            print(f"xt_conv2.shape: {xt.shape}")
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x
    
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################


class Encoder(nn.Module):
    def __init__(self, h,
                 ):
        super().__init__()

        self.h = h
        # h["inter_channels"]
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.out_channels = h["inter_channels"]
        self.num_downsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(Conv1d(1, h["upsample_initial_channel"]// (2 ** len(h["upsample_rates"])), 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(reversed(h["upsample_rates"]), reversed(h["upsample_kernel_sizes"]))):
            self.ups.append(weight_norm(
                Conv1d(h["upsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - i)), h["upsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - i - 1)),
                                k, u, padding= (k - u + 1) // 2)))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups), 0, -1):
            ch = h["upsample_initial_channel"] // (2 ** (i - 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 2 * h["inter_channels"], 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

    def forward(self, x):
        x = x[:,None,:]
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        m, logs = torch.split(x, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(Conv1d(h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(
                ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i), h["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u + 1) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

    def forward(self, x):
        x = self.conv_pre(x)
        print(f"x_conv_pre.shape: {x.shape}")
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            print(f"x_ups{i}.shape: {x.shape}")
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    print(f"x_resblock{i*self.num_kernels+j}.shape: {xs.shape}")
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    print(f"x_resblock{i*self.num_kernels+j}.shape: {xs.shape}")
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        print(f"x_conv_post.shape: {x.shape}")
        x = torch.tanh(x)

        return x
    
    
    
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        """ f0: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshold)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
class Hifigan_Generator(torch.nn.Module):
    def __init__(self, h):
        super(Hifigan_Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = h.mini_nsf
        self.noise_sigma = h.noise_sigma
        
        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2: ]))
            self.upp = int(np.prod(h.upsample_rates[: 2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(
                sampling_rate=h.sampling_rate,
                harmonic_num=8
            )
            self.noise_convs = nn.ModuleList()
        
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))   
        
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
            if not h.mini_nsf:
                if i + 1 < len(h.upsample_rates):  #
                    stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(
                        1, ch, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
    def fastsinegen(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        sines = torch.sin(2 * np.pi * rad)
        return sines
        
    def forward(self, x, f0):
        if self.mini_nsf:
            har_source = self.fastsinegen(f0)
        else:
            har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        print(f"x_conv_pre.shape: {x.shape}")
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x += self.noise_sigma * torch.randn_like(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            print(f"x_ups[{i}].shape: {x.shape}")
            if not self.mini_nsf:
                x_source = self.noise_convs[i](har_source)
                x = x + x_source
            elif i == 1:
                x_source = self.source_conv(har_source)
                x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
    
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################

class Encoder_spec(nn.Module):
    def __init__(self, h,
                 ):
        super().__init__()

        self.h = h
        # h["inter_channels"]
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.in_channels = h["downsample_channels"][0]
        self.out_channels = h["inter_channels"]
        self.num_downsamples = len(h["downsample_rates"])
        self.conv_pre = weight_norm(Conv1d(self.in_channels, self.in_channels, 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        
        self.downs = nn.ModuleList()
        for i, (d, k) in enumerate(zip(h["downsample_rates"], h["downsample_kernel_sizes"])):
            self.downs.append(weight_norm(
                Conv1d(h["downsample_channels"][i], h["downsample_channels"][i+1],
                        k, d, padding= (k - d + 1) // 2)))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h["downsample_channels"][i+1]
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 2 * h["inter_channels"], 7, 1, padding=3))
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

    def forward(self, x):

        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        m, logs = torch.split(x, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs
if __name__ == '__main__':
    h = {
        "upsample_initial_channel": 512,
        "upsample_rates": [8,8,2,2,2],
        "upsample_kernel_sizes": [16,16,4,4,4],
        "resblock": '1',
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
        "inter_channels": 192,
    }
    '''model = Encoder(h)
    print(model)
    x = torch.randn(1, 1024)
    z, m, logs = model(x)
    print(z.shape, m.shape, logs.shape)'''
    
    '''model2 = Generator(h)
    print(model2)
    x = torch.randn(1, 192, 2)
    x = model2(x)
    print(x.shape)'''
    
    '''h_hifigan ={
    "mini_nsf": True,
    "noise_sigma": 0.0,
    "upsample_rates": [ 8, 8, 2, 2, 2 ],
    "upsample_kernel_sizes": [ 16,16, 4, 4, 4 ],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [ 3,7,11 ],
    "resblock_dilation_sizes": [ [ 1,3,5 ], [ 1,3,5 ], [ 1,3,5 ] ],
    "discriminator_periods": [ 3, 5, 7, 11, 17, 23, 37 ],
    "resblock": "1",
    "sampling_rate" : 44100,
    "num_mels" : 128,
    "hop_size" : 512,
    }
    
    model3 = Hifigan_Generator(HParams(h_hifigan))
    print(model3)
    
    x = torch.randn(1, 128, 2)
    f0 = torch.randn( 1, 2)
    x = model3(x, f0)
    print(x.shape)'''
    
    h_spec =  {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
    "upsample_rates": [8,8,2,2,2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16,4,4,4],
    "downsample_rates": [1,1],
    "downsample_kernel_sizes": [5,5],
    "downsample_channels": [1024,768,512]
  }
    print(HParams(**h_spec))
    model3 = Encoder_spec(h_spec)
    print(model3)
    
    x = torch.randn(1, 1024,10)
    z, m, logs = model3(x)
    print(z.shape, m.shape, logs.shape)