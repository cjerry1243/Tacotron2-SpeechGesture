import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

# from common.layers import ConvNorm, LinearNorm
# from common.utils import get_mask_from_lengths, get_mask_from_lengths_window_and_time_step
# from common.gst_beta_vae import GST_BetaVAE

from typing import Optional


def get_mask_from_lengths_window_and_time_step(lengths: torch.Tensor, attention_window_size: int, time_step: int):
    # Mask all initially.
    max_len = torch.max(lengths).item()
    B = len(lengths)
    # mask = torch.cuda.BoolTensor(B, max_len)
    mask = torch.ones(B, max_len, dtype=torch.bool).cuda()
    # mask[:] = 1

    for ii in range(B):
        max_idx = lengths[ii] - 1
        start_idx = int(min([int(max([0, int(time_step - attention_window_size)])), int(max_idx)]))
        # start_idx = int(torch.min(torch.tensor([torch.max(torch.tensor([0, time_step - attention_window_size])), max_idx])))

        end_idx = int(min([int(time_step + attention_window_size), int(max_idx)]))
        if start_idx > end_idx:
            continue
        mask[ii, start_idx:(end_idx+1)] = 0
    return mask




class GST_BetaVAE(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # self.encoder = ReferenceEncoder(hparams)
        self.stl = STL(hparams)

    # def forward(self, inputs: torch.Tensor, return_score=False):
    #     enc_out = self.encoder(inputs)
    #     if return_score:
    #         style_embed, scores, mu, logvar = self.stl(enc_out, return_score=True)
    #         return style_embed, scores, mu, logvar
    #     else:
    #         style_embed, mu, logvar = self.stl(enc_out, return_score=False)
    #         return style_embed, mu, logvar

    def forward(self):
        return

    def infer(self, scores: torch.Tensor): # (h, B, 1, num_tokens)
        style_embed = self.stl.infer(scores)
        return style_embed # (B, 1, num_units)


# class ReferenceEncoder(nn.Module):
#     '''
#     inputs --- [B, Ty/r, n_mels*r]  mels
#     outputs --- [B, ref_enc_gru_size]
#     '''
#     def __init__(self, hparams):
#         super().__init__()
#         K = len(hparams.ref_enc_filters)
#         filters = [1] + hparams.ref_enc_filters
#         convs = [nn.Conv2d(in_channels=filters[i],
#                            out_channels=filters[i + 1],
#                            kernel_size=(3, 3),
#                            stride=(2, 2),
#                            padding=(1, 1)) for i in range(K)]
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i]) for i in range(K)])
#         out_channels = self.calculate_channels(hparams.n_mels, 3, 2, 1, K)
#         self.gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
#                           hidden_size=hparams.token_dim // 2,
#                           batch_first=True)
#         self.n_mels = hparams.n_mels
#
#     def forward(self, inputs):
#         B = inputs.size(0)
#         out = inputs.view(B, 1, -1, self.n_mels)  # [B, 1, Ty, n_mels]
#         for conv, bn in zip(self.convs, self.bns):
#             out = conv(out)
#             out = bn(out)
#             out = F.relu(out)  # [B, 128, Ty//2^K, n_mels//2^K]
#
#         out = out.transpose(1, 2)  # [B, Ty//2^K, 128, n_mels//2^K]
#         T = out.size(1)
#         B = out.size(0)
#         out = out.contiguous().view(B, T, -1)  # [B, Ty//2^K, 128*n_mels//2^K]
#         # self.gru.flatten_parameters()
#         memory, out = self.gru(out)  # out --- [1, B, token_dim//2]
#         return torch.mean(memory, dim=1)
#         # return out.squeeze(0)
#
#     def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
#         for i in range(n_convs):
#             L = (L - kernel_size + 2 * pad) // stride + 1
#         return L


class STL(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # self.embed = nn.Parameter(torch.FloatTensor(hparams.token_num, hparams.token_dim // hparams.num_heads), requires_grad=False)
        self.embed = nn.Parameter(torch.FloatTensor(hparams.token_num, hparams.token_dim // hparams.num_heads), requires_grad=True)
        d_q = hparams.token_dim // 2
        d_k = hparams.token_dim // hparams.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hparams.token_dim, num_heads=hparams.num_heads)
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self):
        return

    # def forward(self, inputs: torch.Tensor, return_score=False):
    #     B = inputs.size(0)
    #     query = inputs.unsqueeze(1)  # [B, 1, token_dim//2]
    #     keys = torch.tanh(self.embed).unsqueeze(0).expand(B, -1, -1)  # [B, token_num, token_dim // num_heads]
    #     if return_score:
    #         style_embed, scores, mu, logvar = self.attention(query, keys, return_score=return_score)
    #         return style_embed, scores, mu, logvar
    #     else:
    #         style_embed, mu, logvar = self.attention(query, keys, return_score=return_score)
    #         return style_embed, mu, logvar

    def infer(self, scores: torch.Tensor):
        keys = torch.tanh(self.embed).unsqueeze(0).expand(1, -1, -1)  # [B, token_num, token_dim // num_heads]
        style_embed = self.attention.infer(scores, keys)
        return style_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

        self.W_query_logvar = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key_logvar = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        # self.W_value_logvar = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self):
        return

    # def forward(self, query: torch.Tensor, key: torch.Tensor, return_score=False):
    #     querys = self.W_query(query)  # [B, 1, num_units]
    #     keys = self.W_key(key)  # [B, num_tokens, num_units]
    #     values = self.W_value(key)
    #
    #     split_size = self.num_units // self.num_heads
    #     querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, B, 1, num_units/h]
    #     keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, B, num_tokens, num_units/h]
    #     values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, B, num_tokens, num_units/h]
    #
    #     mu = torch.matmul(querys, keys.transpose(2, 3))  # [h, B, 1, num_tokens]
    #     mu = mu / (self.key_dim ** 0.5)
    #     # mu = torch.softmax(mu, dim=3)
    #     # mu = torch.sigmoid(mu)
    #     # mu = torch.tanh(mu)
    #
    #     querys = self.W_query_logvar(query)
    #     keys = self.W_key_logvar(key)
    #
    #     querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, B, 1, num_units/h]
    #     keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, B, num_tokens, num_units/h]
    #
    #     logvar = torch.matmul(querys, keys.transpose(2, 3))  # [h, B, 1, num_tokens]
    #     logvar = logvar / (self.key_dim ** 0.5)
    #     # logvar = torch.tanh(logvar)
    #
    #     # re-parameterization
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     scores =  mu + eps * std
    #
    #     out = torch.matmul(scores, values)  # [h, B, 1, num_units/h]
    #     out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, 1, num_units]
    #     if return_score:
    #         return out, scores, mu, logvar
    #     else:
    #         return out, mu, logvar

    def infer(self, scores: torch.Tensor, key: torch.Tensor):
        values = self.W_value(key)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, B, num_tokens, num_units/h]
        out = torch.matmul(scores, values)  # [h, B, 1, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, 1, num_units]
        return out




class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal: torch.Tensor):
        conv_signal = self.conv(signal)
        return conv_signal



class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat: torch.Tensor):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        self.not_so_small_mask = -1000

    def get_alignment_energies(self, query: torch.Tensor, processed_memory: torch.Tensor, attention_weights_cat: torch.Tensor):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state: torch.Tensor, memory: torch.Tensor, processed_memory: torch.Tensor,
                attention_weights_cat: torch.Tensor, mask: torch.Tensor):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x, seed=None):
        # type: (torch.Tensor, Optional[int]) -> torch.Tensor
        if seed is not None:
            torch.manual_seed(seed)
        for linear in self.layers:
            # x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_acoustic_feat_dims,
                         hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim,
                         hparams.n_acoustic_feat_dims,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_acoustic_feat_dims))
            )

    def forward(self, x: torch.Tensor):
        for conv_layer in self.convolutions:
            x1 = conv_layer(x)
            x2 = torch.tanh(x1)
            x = F.dropout(x2, 0.5, self.training)
        x = F.dropout(x1, 0.5, self.training)
        return x


# class Encoder(nn.Module):
#     """Encoder module:
#         - Three 1-d convolution banks
#         - Bidirectional LSTM
#     """
#     def __init__(self, hparams):
#         super(Encoder, self).__init__()
#
#         self.prenet = Prenet(hparams.n_symbols,
#                              [hparams.symbols_embedding_dim,
#                               hparams.symbols_embedding_dim])
#
#         convolutions = []
#         for _ in range(hparams.encoder_n_convolutions):
#             conv_layer = nn.Sequential(
#                 ConvNorm(hparams.encoder_embedding_dim,
#                          hparams.encoder_embedding_dim,
#                          kernel_size=hparams.encoder_kernel_size, stride=1,
#                          padding=int((hparams.encoder_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='relu'),
#                 nn.BatchNorm1d(hparams.encoder_embedding_dim))
#             convolutions.append(conv_layer)
#         self.convolutions = nn.ModuleList(convolutions)
#
#         self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
#                             int(hparams.encoder_embedding_dim / 2), 1,
#                             batch_first=True, bidirectional=True)
#         self.lstm.flatten_parameters()
#     def forward(self, x, input_lengths):
#         # x: (B, D, T) -> (B, T, D) -> (B, D, T)
#         x = self.prenet(x.transpose(1, 2)).transpose(1, 2)
#
#         for conv in self.convolutions:
#             x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#
#         x = x.transpose(1, 2)
#
#         # pytorch tensor are not reversible, hence the conversion
#         #input_lengths = input_lengths.cpu().numpy()
#         x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
#
#         outputs, _ = self.lstm(x)
#
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#
#         return outputs
#
#     def inference(self, x, seed=None):
#         # x: (B, D, T) -> (B, T, D) -> (B, D, T)
#         x = self.prenet(x.transpose(1, 2), seed).transpose(1, 2)
#
#         for conv in self.convolutions:
#             x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#
#         x = x.transpose(1, 2)
#
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(x)
#
#         return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_acoustic_feat_dims = hparams.n_acoustic_feat_dims
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.spk_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.attention_window_size = hparams.attention_window_size
        self.prenet = Prenet(hparams.n_acoustic_feat_dims,
                             [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.spk_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim + hparams.spk_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.spk_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.spk_embedding_dim,
            hparams.n_acoustic_feat_dims)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.spk_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

        self.attention_hidden = torch.tensor([])
        self.attention_cell = torch.tensor([])
        self.decoder_hidden = torch.tensor([])
        self.decoder_cell = torch.tensor([])
        self.attention_weights = torch.tensor([])
        self.attention_weights_cum = torch.tensor([])
        self.attention_context = torch.tensor([])

        self.memory = torch.tensor([])
        self.processed_memory = torch.tensor([])
        # self.mask = Optional[torch.Tensor]
        self.mask = torch.tensor([])

    def forward(self):
        return

    def get_go_frame(self, memory: torch.Tensor):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        #decoder_input = Variable(memory.data.new(
        #    B, self.n_acoustic_feat_dims).zero_())
        decoder_input = torch.zeros(B, self.n_acoustic_feat_dims).to(memory.device)
        return decoder_input

    def initialize_decoder_states(self, memory: torch.Tensor):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        """
        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())
        """
        self.attention_hidden = torch.zeros(B, self.attention_rnn_dim).to(memory.device)
        self.attention_cell = torch.zeros(B, self.attention_rnn_dim).to(memory.device)
        self.decoder_hidden = torch.zeros(B, self.decoder_rnn_dim).to(memory.device)
        self.decoder_cell = torch.zeros(B, self.decoder_rnn_dim).to(memory.device)
        self.attention_weights = torch.zeros(B, MAX_TIME).to(memory.device)
        self.attention_weights_cum = torch.zeros(B, MAX_TIME).to(memory.device)
        self.attention_context = torch.zeros(B, self.encoder_embedding_dim).to(memory.device)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory).to(memory.device)
        # self.mask = mask

    def parse_decoder_outputs(self, acoustic_outputs: torch.Tensor, gate_outputs: torch.Tensor, alignments: torch.Tensor):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        acoustic_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        acoustic_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1)
        # alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1)
        # gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_acoustic_feat_dims) -> (B, T_out, n_acoustic_feat_dims)
        acoustic_outputs = acoustic_outputs.transpose(0, 1).contiguous()
        # acoustic_outputs = torch.stack(acoustic_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        acoustic_outputs = acoustic_outputs.view(
            acoustic_outputs.size(0), -1, self.n_acoustic_feat_dims)
        # (B, T_out, n_acoustic_feat_dims) -> (B, n_acoustic_feat_dims, T_out)
        acoustic_outputs = acoustic_outputs.transpose(1, 2)

        return acoustic_outputs, gate_outputs, alignments

    def decode(self, decoder_input: torch.Tensor, attention_windowed_mask: Optional[torch.Tensor]=None):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        tmp = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = tmp[0]
        self.attention_cell = tmp[1]
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        if attention_windowed_mask is None:
            tmp = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)
            self.attention_context = tmp[0]
            self.attention_weights = tmp[1]
        else:
            tmp = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, attention_windowed_mask)
            self.attention_context = tmp[0]
            self.attention_weights = tmp[1]

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        tmp = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = tmp[0]
        self.decoder_cell = tmp[1]
        # self.decoder_hidden, self.decoder_cell = self.decoder_rnn2(
        #     self.decoder_hidden, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def inference(self, memory: torch.Tensor, memory_lengths: torch.Tensor):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        acoustic_outputs: acoustic outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory)

        acoustic_outputs, gate_outputs, alignments = [], [], []
        for _ in range(int(memory_lengths)):
            decoder_input = self.prenet(decoder_input)

            if self.attention_window_size is not None:
                time_step = len(acoustic_outputs)
                attention_windowed_mask = get_mask_from_lengths_window_and_time_step(memory_lengths, self.attention_window_size, time_step)
            else:
                attention_windowed_mask = None

            acoustic_output, gate_output, alignment = self.decode(decoder_input, attention_windowed_mask)
            acoustic_outputs += [acoustic_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            decoder_input = acoustic_output

        acoustic_outputs = torch.stack(acoustic_outputs)
        gate_outputs = torch.stack(gate_outputs)
        alignments = torch.stack(alignments)
        acoustic_outputs, gate_outputs, alignments = self.parse_decoder_outputs(acoustic_outputs, gate_outputs, alignments)

        return acoustic_outputs, gate_outputs, alignments


class Tacotron2_gst_beta_vae_scriptmodule(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2_gst_beta_vae_scriptmodule, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_acoustic_feat_dims = hparams.n_acoustic_feat_dims
        # self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.gst = GST_BetaVAE(hparams)
        self.num_tokens = hparams.token_num
        self.num_heads = hparams.num_heads

    def forward(self):
        return

    def inference_from_encoder(self, encoder_outputs: torch.Tensor, scores: torch.Tensor):
        token = self.gst.infer(scores).repeat(1, encoder_outputs.shape[1], 1)
        encoder_outputs = torch.cat((encoder_outputs, token), dim=-1)
        acoustic_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs, torch.tensor([encoder_outputs.shape[1]], dtype=torch.long).cuda())

        acoustic_outputs_postnet = self.postnet(acoustic_outputs)
        acoustic_outputs_postnet = acoustic_outputs + acoustic_outputs_postnet

        return acoustic_outputs, acoustic_outputs_postnet, gate_outputs, alignments
