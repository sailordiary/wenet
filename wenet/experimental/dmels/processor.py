import torch
import torchaudio


def compute_melspectrogram(sample,
                           sample_rate=24000,
                           n_fft=1024,
                           hop_length=256,
                           n_mels=100,
                           center=False,
                           power=1):
    waveform = sample['waveform']
    specgram = torchaudio.functional.spectrogram(
        waveform,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        power=power,
        pad=0,
        n_fft=n_fft,
        center=center,
        normalized=False,
        onesided=True)
    mel_scale = torchaudio.functional.melscale_fbanks(
        n_fft // 2 + 1,
        0,
        float(sample_rate // 2),
        n_mels,
        sample_rate,
        mel_scale="htk",
    )

    mel_specgram = torch.matmul(specgram.transpose(-1, -2),
                                mel_scale).transpose(-1, -2)
    # sample['mel_specgram'] = mel_specgram
    sample['mel_specgram'] = torch.log(torch.clip(mel_specgram, min=1e-5))
    return sample


if __name__ == '__main__':
    # TODO: remove in future
    def create_codebook(m, M, K):
        """
        创建码本C
        """
        delta = (M - m) / (2**K)
        codebook = torch.arange(m, M, delta)
        return codebook

    def discretize(M, codebook):
        """
        将mel滤波器组表示离散化为bin索引
        """
        M_flat = M.flatten()
        codebook_indices = torch.argmin(
            torch.abs(M_flat.unsqueeze(1) - codebook.unsqueeze(0)), dim=1)
        discretized_M = codebook_indices.view(M.shape)
        return discretized_M

    def reconstruct(discretized_M, codebook):
        """
        从bin索引重建mel滤波器组表示
        """
        reconstructed_M = codebook[discretized_M]
        return reconstructed_M

    def get_m_M():
        speech, sr = torchaudio.load('test.wav')
        speech_mels = compute_melspectrogram({"waveform":
                                              speech})['mel_specgram']

        music, sr = torchaudio.load('test.music.wav')
        music_mels = compute_melspectrogram({"waveform":
                                             music})['mel_specgram']

        print(music_mels.shape, speech_mels.shape)
        all_samples = torch.cat((speech_mels, music_mels), dim=-1)
        m = torch.min(all_samples)
        M = torch.max(all_samples)
        return m, M

    # vocoder_ckpt = torch.load("vocoder.pth", map_location='cpu')
    # vocoder = UnivNetGenerator()
    # vocoder.load_state_dict(vocoder_ckpt['model_g'])
    # vocoder.eval()
    wav = 'test.music.wav'
    wav = 'test.wav'
    waveform, sr = torchaudio.load(wav)
    # waveform = torchaudio.functional.resample(wave, sr, 24000)
    mels = compute_melspectrogram({"waveform": waveform})['mel_specgram']
    # mels = MelSpectrogramFeatures(padding='center')(waveform)

    # wav = vocoder.inference(mels)
    from vocos import Vocos

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    wav = vocos.decode(mels)
    torchaudio.save('out.2.wav', wav, sample_rate=24000, bits_per_sample=16)

    K = 4
    # m = torch.min(mels)
    # M = torch.max(mels)
    m, M = get_m_M()
    codebook = create_codebook(m, M, K)
    discretized_M = discretize(mels, codebook)
    print(discretized_M.shape)
    reconstructed_M = reconstruct(discretized_M, codebook)
    print(reconstructed_M, reconstructed_M.shape)
    wav = vocos.decode(reconstructed_M)
    torchaudio.save('dmels.music.2.wav',
                    wav,
                    sample_rate=24000,
                    bits_per_sample=16)
