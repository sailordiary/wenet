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

    def __init__(self, mel_min: float, mel_max: float, k=4) -> None:
        super().__init__()
        self.register_buffer('min', torch.tensor(mel_min, requires_grad=False))
        self.register_buffer('max', torch.tensor(mel_max, requires_grad=False))
        self.register_buffer('codebook', _create_codebook(mel_min, mel_max, k))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return quantize(x, self.codebook)

    def decode(self, codes: torch.Tensor):
        return dequantize(codes, self.codebook)
