import torch
import torch.nn as nn
from layers.pyraformer.Layers import EncoderLayer, Decoder, Predictor
from layers.pyraformer.Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.pyraformer.Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from layers.pyraformer.embed import DataEmbedding, CustomEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.input_size = opt.seq_len
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        self.decoder_type = opt.pyraformer_decoder_type
        
        if self.decoder_type == 'attention':
            mask, self.all_size = get_mask(self.input_size, opt.window_size, opt.inner_size)
        else:
            mask, self.all_size = get_mask(self.input_size+1, opt.window_size, opt.inner_size)
        self.register_buffer('mask', mask)

        if self.decoder_type == 'FC':
            indexes = refer_points(self.all_size, opt.window_size)
            self.register_buffer('indexes', indexes)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if self.decoder_type == 'FC' else 0
            q_k_mask = get_q_k(self.input_size + padding, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_heads, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(opt.e_layers)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_heads, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.e_layers)
                ])

        if opt.pyraformer_embed_type == 'CustomEmbedding':
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.predict_step = opt.pred_len
        self.d_model = opt.d_model
        self.input_size = opt.seq_len
        self.decoder_type = opt.pyraformer_decoder_type
        self.channels = opt.enc_in

        self.encoder = Encoder(opt)
        if self.decoder_type == 'attention':
            mask = get_subsequent_mask(self.input_size, opt.window_size, self.predict_step, opt.truncate)
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.enc_in)
        elif self.decoder_type == 'FC':
            self.predictor = Predictor(4 * opt.d_model, self.predict_step * opt.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, pretrain=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            # Add a predict token into the history sequence
            predict_token = torch.zeros(x_enc.size(0), 1, x_enc.size(-1), device=x_enc.device)
            x_enc = torch.cat([x_enc, predict_token], dim=1)
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, 0:1, :]], dim=1)

            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

        return pred
