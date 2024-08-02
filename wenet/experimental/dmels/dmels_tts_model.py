from typing import Dict, Optional
import torch
from wenet.LLM.causallm_model import CausalLM
from wenet.LLM.decoder import DecoderOnly

from wenet.experimental.dmels.dmels_quantizer import DmelsQuantizer
from wenet.utils.common import IGNORE_ID
from wenet.utils.mask import make_non_pad_mask, subsequent_mask


class DmelsTTSModel(CausalLM):

    def __init__(self,
                 vocab_size: int,
                 mel_bins: int,
                 decoder: DecoderOnly,
                 quantizer: DmelsQuantizer,
                 special_tokens: dict,
                 tie_word_embedding: bool = False,
                 linear_bias: bool = False,
                 ignore_id: int = IGNORE_ID,
                 lsm_weight: float = 0,
                 reduction: str = 'mean') -> None:
        super().__init__(vocab_size, decoder, special_tokens,
                         tie_word_embedding, linear_bias, ignore_id,
                         lsm_weight, reduction)

        self.quantizer = quantizer

        self.text_embed = torch.nn.Embedding(vocab_size, decoder.hidden_size)

        bits = self.quantizer.bits
        self.speech_tokens_embed = torch.nn.Embedding(
            bits,
            # TODO: change later
            32)
        # TODO(Mddct): why encoder.output_size?
        self.speech_linear = torch.nn.Linear(32 * mel_bins,
                                             decoder.hidden_size)

        self.speech_parallel_head = torch.nn.parameter.Parameter(
            torch.empty(decoder.hidden_size, mel_bins, bits),
            requires_grad=False,
        )
        torch.nn.init.xavier_uniform_(self.speech_parallel_head)
        # TODO: bias

    def forward(self, batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        # 1 speech tokens
        speech = batch['feats'].to(device)  # (B, T, D)
        B, T, _ = speech.shape
        speech_tokens = self.quantizer(speech.transpose(1, 2)).transpose(
            1, 2)  # (B,T,D)

        # TODO(Mddct): span mask, can work with w2vec mask?
        embed = self.speech_tokens_embed(speech_tokens)  # (B,T,D,d)
        embed = embed.view(B, T, -1)  # (B, T, Dxd)
        embed = self.speech_linear(embed)  # (B, T, encoder_dim)

        # 2 text tokens
        text = batch['text'].to(device)
        fake_text_pos = batch['fake_text_pos'].to(device)

        token_embed = self.text_embed(text)  # (B, T,D)
        # TODO: make it right, merge text and speech token embed
        token_embed[fake_text_pos] = embed
        lengths = batch['lengths'].to(device)
        seq_mask = make_non_pad_mask(lengths=lengths)
        causal_mask = subsequent_mask(seq_mask.size(-1),
                                      device=seq_mask.device).unsqueeze(
                                          0)  # (1,L,L)
        att_mask = causal_mask & seq_mask  # (B, L, L)
        decoder_out = self.out(self.decoder(token_embed,
                                            att_mask)[0])  # (B, L, vocab_size)

        out = torch.einsum("btd,dmn->btmn", decoder_out,
                           self.speech_parallel_head)
        labels = batch['labels'].to(device)  # [B,T,M]

        bits = out.size()[-1]
        loss = torch.nn.functional.cross_entropy(out.view(-1, bits),
                                                 labels.view(-1))

        # TODO: topk acc
        return {
            "loss": loss,
        }
