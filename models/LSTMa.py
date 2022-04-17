import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Model(nn.Module):
    # Modification of LSTM for multistep forcasting
    def __init__(self, args):
        super(Model, self).__init__()
        configs = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len

        self.encoder = Seq2SeqEncoder(self.seq_len, configs.d_model, configs.d_model, args.e_layers)
        self.decoder = Seq2SeqDecoder(configs.c_out, configs.d_model, configs.d_model, args.d_layers)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        batch_size, L, D = enc_out.shape
        enc_out = enc_out.permute(1, 0, 2) # L, Batch, D
        enc_out, hidden = self.encoder(enc_out)

        max_len = self.pred_len
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_size).to(x_enc.device)
        hidden = hidden[:self.decoder.n_layers]

        output = x_dec[:, self.label_len, :]  # trg[0, :]  # sos
        time = x_mark_dec[:, self.label_len, :]

        for t in range(max_len):
            dec_out = self.dec_embedding(output.unsqueeze(1), time.unsqueeze(1))
            dec_out = dec_out.permute(1, 0, 2) # L, B, D
            output, hidden, attn_weights = self.decoder(dec_out, hidden, enc_out)
            outputs[t] = output
            is_teacher = random.random() < 0.0
            output = x_dec[:, self.label_len+t, :] if is_teacher else output
            time = x_mark_dec[:, self.label_len+t, :]

        return outputs.permute(1, 0, 2).contiguous()

class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Seq2SeqEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.embed = embedding_layer
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, embedded, hidden=None):
        # embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Seq2SeqDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers=1, dropout=0.2):
        super(Seq2SeqDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        #self.embed = embedding_layer
        #self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, embedded, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        # embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        # embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        return output, hidden, attn_weights


class Seq2SeqDynamic(nn.Module):
    def __init__(self, input_len, target_len, n_emb, emb_dim, hdim, n_layer, device):
        super(Seq2SeqDynamic, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(n_emb, emb_dim)
        self.encoder = Seq2SeqEncoder(input_len, self.embedding, emb_dim, hdim, n_layer)
        self.decoder = Seq2SeqDecoder(n_emb, self.embedding, emb_dim, hdim, n_layer)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        src_emb = self.embedding(src)
        encoder_output, hidden = self.encoder(src_emb)
        hidden = hidden[:self.decoder.n_layers]
        output = src[-1, :]  # trg[0, :]  # sos
        for t in range(max_len):
            output = self.embedding(output).unsqueeze(0)
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            output = trg.data[t] if is_teacher else top1
        return outputs