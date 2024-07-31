import torch
import torchaudio


def compute_melspectrogram(sample,
                           sample_rate=24000,
                           n_fft=1024,
                           hop_length=256,
                           n_mels=100,
                           center=False,
                           power=1):
    waveform = sample['wav']
    assert waveform.size(0) == 1
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
    mel_specgram = torch.log(torch.clip(mel_specgram, min=1e-5))
    sample['feat'] = mel_specgram.transpose(1, 2).squeeze(0)
    return sample


if __name__ == '__main__':
    pass
    # wav = 'test.music.wav'
    # wav = 'test.wav'
    # waveform, sr = torchaudio.load(wav)
    # # waveform = torchaudio.functional.resample(wave, sr, 24000)
    # mels = compute_melspectrogram({"waveform": waveform})['mel_specgram']
    # # mels = MelSpectrogramFeatures(padding='center')(waveform)

    # # wav = vocoder.inference(mels)
    # from vocos import Vocos

    # vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    # wav = vocos.decode(mels)
    # torchaudio.save('out.2.wav', wav, sample_rate=24000, bits_per_sample=16)

    # K = 4
    # # m = torch.min(mels)
    # # M = torch.max(mels)
    # m, M = get_m_M()
    # codebook = create_codebook(m, M, K)
    # discretized_M = discretize(mels, codebook)
    # print(discretized_M.shape)
    # reconstructed_M = reconstruct(discretized_M, codebook)
    # print(reconstructed_M, reconstructed_M.shape)
    # wav = vocos.decode(reconstructed_M)
    # torchaudio.save('dmels.music.2.wav',
    #                 wav,
    #                 sample_rate=24000,
    #                 bits_per_sample=16)
