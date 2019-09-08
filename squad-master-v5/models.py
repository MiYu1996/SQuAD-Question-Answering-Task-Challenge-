"""Top-level model classes.

Authors:
    Louis Miao Miao Huang
    Jeremy Mi Yu
    Carson Yu Tian Zhao
    Chris Chute (chute@stanford.edu)
"""

import layers
# import layersQANet as qa
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, char_vectors, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.hidden_size = hidden_size

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    char_vectors = char_vectors)   # added character vectors

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)


        ### start our code:
        self.selfattention = layers.SelfAttention(input_size = 8 * hidden_size,
                                                  hidden_size=hidden_size,
                                                  dropout = 0.2)

        ### end our code
        self.linear = nn.Linear(in_features = 8*self.hidden_size, out_features = 2*self.hidden_size, bias=True)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=4,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    # def forward(self, cw_idxs, qw_idxs):    # orig
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)


        # Word/Char embeddings layer
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # RNN encoder layer
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # BiDAF Attention layer
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # change: remove modeling layer from baseline
            # replace with self-matching attention layer
        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
        mod = self.linear(att) + mod

        ### start our code:
        mod = self.selfattention(mod)

        ### end our code

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

