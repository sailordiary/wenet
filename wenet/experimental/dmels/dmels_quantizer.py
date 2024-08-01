import torch


def _create_codebook(m, M, K):
    delta = (M - m) / (2**K)
    codebook = torch.arange(m, M, delta)
    return codebook


def quantize(M, codebook):
    M_flat = M.flatten()
    codebook_indices = torch.argmin(
        torch.abs(M_flat.unsqueeze(1) - codebook.unsqueeze(0)), dim=1)
    discretized_M = codebook_indices.view(M.shape)
    return discretized_M


def dequantize(discretized_M, codebook):
    reconstructed_M = codebook[discretized_M]
    return reconstructed_M


class DmelsQuantizer(torch.nn.Module):
    """ https://arxiv.org/html/2407.15835v1
    """

    def __init__(self, mel_min: float, mel_max: float, k=4) -> None:
        super().__init__()
        self.register_buffer('min', torch.tensor(mel_min, requires_grad=False))
        self.register_buffer('max', torch.tensor(mel_max, requires_grad=False))
        self.register_buffer('codebook', _create_codebook(mel_min, mel_max, k))

        self.k = k

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return quantize(x, self.codebook)

    def decode(self, codes: torch.Tensor):
        return dequantize(codes, self.codebook)

    @property
    def bits(self):
        return 2**self.k


if __name__ == '__main__':
    from wenet.experimental.dmels.processor import compute_melspectrogram
    import torchaudio
    wav_file = 'cat.wav'
    out_wav_file = 'cat.out.wav'
    waveform, sr = torchaudio.load(wav_file)
    waveform = waveform[:1]
    waveform = torchaudio.functional.resample(waveform, sr, 24000)
    sr = 24000
    mels = compute_melspectrogram({'wav': waveform})['feat']
    print(mels.shape)
    print(mels)

    print('min:', torch.min(mels), 'max:', torch.max(mels))
    quantizer = DmelsQuantizer(torch.min(mels), torch.max(mels))
    print(quantizer.codebook)

    tokens = quantizer(mels)
    print(tokens, tokens.shape)

    mels_gen = quantizer.decode(tokens)
    print(mels_gen)

    # wav = vocoder.inference(mels)
    from vocos import Vocos

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    wav = vocos.decode(mels_gen)
    torchaudio.save(out_wav_file, wav, sample_rate=24000, bits_per_sample=16)
