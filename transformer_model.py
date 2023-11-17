import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

train_batch_size = 128
d_model = 512  # data dimension
d_data = 100 # number of historical value of each stock
d_q = d_k = d_v = 64  # wq=wk=wv
# ffn_hidden = 2048  # feedforward layer
n_heads = 8 # multi heads
n_layer = 6 # encoder layers


class TransformerEncoderOnly(nn.Module):
    def __init__(self):
        super(TransformerEncoderOnly, self).__init__()
        self.encoder=Encoder()
        self.predict_ln=nn.Linear(d_model,1) # predict the value

    def forward(self, inputs):
        # [batch_size, input_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(inputs)
        pred_outputs=torch.squeeze(self.predict_ln(enc_outputs),-1) # [batch_size, input_len]
        return pred_outputs, enc_self_attns


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        # Q: [batch_size, n_heads, len_q, d_q]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v, d_v]
        scores = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_q = nn.Linear(d_model, d_q * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(d_v*n_heads, d_model)
        # self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_q, input_k, input_v):
        # [batch_size, input_len, d_model], input_len means the number of words in the sentence
        batch_size = input_q.size(0)
        # Q: [batch_size, n_heads, len_q, d_q]
        Q = self.W_q(input_q).view(batch_size, -1, n_heads, d_q).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_k(input_k).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_v(input_v).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        context, attn = ScaleDotProductAttention()(Q, K, V)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, d_v*n_heads)

        output = self.fc(context)
        # return self.layernorm(output+residual), attn
        return output, attn


class Row_wiseFeedforward(nn.Module):
    def __init__(self):
        super(Row_wiseFeedforward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention()
        self.ln0 = nn.LayerNorm(d_model)
        self.rff = Row_wiseFeedforward()
        self.ln1 = nn.LayerNorm(d_model)

    def forward(self, enc_inputs):
        # [batch_size, input_len, d_model]
        residual0 = enc_inputs
        msa_outputs, attn = self.self_attention(enc_inputs, enc_inputs, enc_inputs)
        ln0_outputs = self.ln0(residual0+msa_outputs)
        residual1 = ln0_outputs
        rff_outputs = self.rff(ln0_outputs)
        enc_outputs = self.ln1(rff_outputs+residual1)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.ln = nn.Linear(d_data, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer for _ in range(n_layer)])

    def forward(self, inputs):
        # [batch_size, input_len, d_data]
        enc_inputs = self.ln(inputs)
        enc_outputs = enc_inputs
        enc_self_attns=[]
        for layer in self.enc_layers:
            enc_outputs,enc_self_attn=layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns