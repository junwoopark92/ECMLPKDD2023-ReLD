"""
N-BEATS Model.
"""
from typing import Tuple

import numpy as np
import torch as t

# This implementation is based on ElementAI N-BEATS code implementation
# tommy.dm.kim@gmail.com

class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.
        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor) -> t.Tensor:
        residuals = x
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast)
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


class Model(t.nn.Module):
    '''
    Nbeats
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        block_fns = {
            "interpretable": self.get_interpretable_block,
            "generic": self.get_generic_block,
        }
        block_type = configs.nbeats_mode
        self.nbeats_layer = block_fns[block_type](configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc:(B, L, D) => (B,L)
        x_enc = x_enc.squeeze(2)
        pred = self.nbeats_layer(x_enc)
        # pred:(B, L) => (B, L, D)
        pred = pred.unsqueeze(2)
        return pred

    def get_interpretable_block(self, configs):
        trend_block = NBeatsBlock(
            input_size = configs.seq_len,
            theta_size=2 * (configs.degree_of_polynomial + 1),
            basis_function=TrendBasis(
                degree_of_polynomial=configs.degree_of_polynomial,
                backcast_size=configs.seq_len,
                forecast_size=configs.pred_len),
            layers=configs.trend_layers,
            layer_size=configs.trend_layer_size
        )

        seasonality_block = NBeatsBlock(
            input_size=configs.seq_len,
            theta_size=4 * int(np.ceil(configs.num_of_harmonics / 2 * configs.pred_len) - (configs.num_of_harmonics - 1)),
            basis_function=SeasonalityBasis(
                harmonics=configs.num_of_harmonics,
                backcast_size=configs.seq_len,
                forecast_size=configs.pred_len
            ),
            layers=configs.seasonality_layers,
            layer_size=configs.seasonality_layer_size
        )

        block = NBeats(
            t.nn.ModuleList(
                [trend_block for _ in range(configs.trend_blocks)] +
                [seasonality_block for _ in range(configs.seasonality_blocks)])
        )

        return block

    def get_generic_block(self, configs):
        block = NBeats(
            t.nn.ModuleList(
                [NBeatsBlock(
                    input_size=configs.seq_len,
                    theta_size=configs.seq_len + configs.pred_len,
                    basis_function=GenericBasis(
                        backcast_size=configs.seq_len,
                        forecast_size=configs.pred_len),
                    layers=configs.nb_layers,
                    layer_size=configs.d_model
                ) for _ in range(configs.nb_stacks)]))
        return block