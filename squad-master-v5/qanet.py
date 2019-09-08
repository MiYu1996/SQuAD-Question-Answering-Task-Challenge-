"""Top-level model classes.

Authors:
    Louis Miao Miao Huang
    Jeremy Mi Yu
    Carson Yu Tian Zhao
"""

import layers
# import layersQANet as qa
import torch
import torch.nn as nn

class QANet(nn.Module):
    """Implementation of QANet:
    Combining local convolution with global self-attention

    Structure:
        - Input embedding layer
        - Embedding encoder layer
        - Context-Query attention layer
        - Model encoder layer
        - Output layer

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        char_vectors (torch.Tensor):
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, char_vectors, drop_prob=0.):
        super(QANet, self).__init__()

        self.D = 96
        self.Nh = 8
        self.Dword = 300
        self.Dchar = 64
        self.batch_size = 16
        self.dropout = 0.1
        self.dropout_char = 0.05
        self.Dk = self.D // self.Nh
        self.Dv = self.D // self.Nh
        self.D_cq_att = self.D * 4
        self.Lc = 400   # Limit length for paragraph
        self.Lq = 50    # Limit length for answers

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    char_vectors = char_vectors)   # added character vectors

        self.c_conv = qa.DepthwiseSeparableConv(Dword+Dchar, D, 5)
        self.q_conv = qa.DepthwiseSeparableConv(Dword+Dchar, D, 5)
        self.c_enc = qa.EncoderBlock(conv_num=4, ch_num=D, k=7, length=Lc)
        self.q_enc = qa.EncoderBlock(conv_num=4, ch_num=D, k=7, length=Lq)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.resize = qa.DepthwiseSeparableConv(D * 4, D, 5)

        enc_blk = qa.EncoderBlock(conv_num=2, ch_num=D, k=5, length=Lc)
        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)

        self.output = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # Input embedding layer
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)


        # Embedding encoder layer
        c = self.context_conv(c_emb)
        q = self.question_conv(q_emb)
        c_e = self.c_emb_enc(c, c_mask)
        q_e = self.q_emb_enc(q, q_mask)


        # Context-Query attention layer
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        m1 = self.cq_resizer(att)

        # Model encoder layer
        for enc in self.model_enc_blks:
            m1 = enc(m1, c_mask)
        m2 = m1
        for enc in self.model_enc_blks:
            m2 = enc(m2, c_mask)
        m3 = m2
        for enc in self.model_enc_blks:
            m3 = enc(m3, c_mask)

        # Output layer
        out = self.output(m1, m2, m3, c_mask)
        return out


