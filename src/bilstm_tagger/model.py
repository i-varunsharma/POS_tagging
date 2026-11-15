# src/bilstm_tagger/model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMTagger(nn.Module):
    def __init__(self, n_words, n_chars, n_tags,
                 word_dim=100, char_dim=30, char_filters=50,
                 hidden=128, layers=1, dropout=0.33,
                 use_char=True, pad_idx=0):
        super().__init__()
        self.use_char = use_char
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=pad_idx)
        in_dim = word_dim

        if use_char:
            self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=pad_idx)
            self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
            in_dim += char_filters

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(2 * hidden, n_tags)

    def forward(self, words, chars, lengths):
        x = self.word_emb(words)                               # (B, T, W)

        if self.use_char:
            B, T, L = chars.shape
            c = self.char_emb(chars.view(B * T, L))            # (B*T, L, C)
            c = self.char_cnn(c.transpose(1, 2))               # (B*T, F, L)
            c = torch.relu(c).max(dim=2).values                # (B*T, F)
            x = torch.cat([x, c.view(B, T, -1)], dim=-1)       # (B, T, W+F)

        x = self.drop(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=words.size(1))
        return self.out(self.drop(out))                        # (B, T, n_tags)