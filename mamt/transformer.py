import math
import torch
import torch.nn as nn
from .audio_config import MAMT_AudioConfig
from .token_config import MAMT_TokenConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class MAMT_Transformer(nn.Module):
    def __init__(
        self,
        n_mels: int = MAMT_AudioConfig.NMEL,
        d_model: int = 512,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.2,
        pad_id: int = MAMT_TokenConfig.PAD_ID,
        vocab_size: int = MAMT_TokenConfig.VOCAB_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        ## Encoder
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        reduced_f = 32 # 256 / 2 / 2 / 2
        in_dim = 128 * reduced_f

        self.enc_in = nn.Linear(in_dim, d_model)
        self.enc_pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # onset/offset head
        self.onset_head  = nn.Linear(d_model, 1)
        self.offset_head = nn.Linear(d_model, 1)

        ## Decoder
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dec_pos = PositionalEncoding(d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, mel, mel_mask, dec_inp):
        """
        mel:             (B, T, n_mels)\n
        mel_mask:        (B, T) True=valid, False=pad\n
        dec_inp:         (B, L)\n
        return:
          logits:        (B, L, V)
          onset_logits:  (B, T)
          offset_logits: (B, T)
        """
        B, T, F = mel.shape

        ## Encoder
        x = mel.unsqueeze(1) # (B, 1, T, F)
        x = self.cnn2d(x)    # (B, 64, T, F')

        B, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T2, C * F2)

        x = self.enc_in(x)  # (B, T, d_model)
        x = self.enc_pos(x) # (B, T, d_model)

        src_key_padding_mask = ~mel_mask # True=pad
        enc_out = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        ) # (B, T, d_model)

        onset_logits  = self.onset_head(enc_out).squeeze(-1)  # (B, T)
        offset_logits = self.offset_head(enc_out).squeeze(-1) # (B, T)

        ## Decoder
        y = self.tok_emb(dec_inp) * math.sqrt(self.d_model) # (B, L, d_model)
        y = self.dec_pos(y)

        tgt_key_padding_mask = (dec_inp == self.pad_id)

        L = dec_inp.size(1)
        causal = torch.triu(
            torch.ones(L, L, device=dec_inp.device, dtype=torch.bool),
            diagonal=1
        ) # (L, L), upper-tri=True

        dec_out = self.decoder(
            tgt=y,
            memory=enc_out,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        ) # (B, L, d_model)

        logits = self.out_proj(dec_out) # (B, L, vocab_size)

        return logits, onset_logits, offset_logits