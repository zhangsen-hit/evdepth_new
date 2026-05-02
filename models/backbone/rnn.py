import math
from typing import Optional, Tuple

import torch as th
import torch.nn as nn


def _gn_num_groups(channels: int, preferred: int = 8) -> int:
    g = min(preferred, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def _chrono_ifg_bias(bias: th.Tensor, dim: int, T_max: int, coupled: bool = True) -> None:
    """Chrono-style init for 3*dim IFG block: forget ~ log U(1,T), optional input = -forget."""
    if T_max is None or T_max < 2:
        return
    with th.no_grad():
        b_f = bias[dim : 2 * dim]
        b_f.uniform_(math.log(1.5), math.log(float(T_max)))
        if coupled:
            bias[:dim].copy_(-b_f)


class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.,
                 T_max_chrono_init: Optional[int] = None):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim
        self.T_max_chrono_init = T_max_chrono_init

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

        if T_max_chrono_init is not None and self.conv1x1.bias is not None:
            _chrono_ifg_bias(self.conv1x1.bias, dim, T_max_chrono_init, coupled=True)

    def forward(self, x: th.Tensor, h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None) \
            -> Tuple[th.Tensor, th.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = th.zeros_like(x)
            cell = th.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)
        xh = th.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = th.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = th.sigmoid(gates)
        forget_gate, input_gate, output_gate = th.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(th.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * th.tanh(c_t)

        return h_t, c_t


class StandardConvLSTM2d(nn.Module):
    """Standard ConvLSTM: single 3x3 conv on concat(x, h) -> 4*dim gates (i, f, o, cell)."""

    def __init__(
        self,
        dim: int,
        cell_update_dropout: float = 0.,
        T_max_chrono_init: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        xh_dim = dim * 2
        gates_dim = dim * 4
        self.gate_conv = nn.Conv2d(
            in_channels=xh_dim,
            out_channels=gates_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)
        if T_max_chrono_init is not None and self.gate_conv.bias is not None:
            _chrono_ifg_bias(self.gate_conv.bias, dim, T_max_chrono_init, coupled=True)

    def forward(self, x: th.Tensor, h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None):
        if h_and_c_previous is None:
            h_tm1 = th.zeros_like(x)
            c_tm1 = th.zeros_like(x)
        else:
            h_tm1, c_tm1 = h_and_c_previous

        xh = th.cat((x, h_tm1), dim=1)
        mix = self.gate_conv(xh)
        i_gate, forget_gate, output_gate, cell_input = th.tensor_split(mix, 4, dim=1)
        input_gate = th.sigmoid(i_gate)
        forget_gate = th.sigmoid(forget_gate)
        output_gate = th.sigmoid(output_gate)
        cell_input = self.cell_update_dropout(th.tanh(cell_input))
        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * th.tanh(c_t)
        return h_t, c_t


class DWSConvSTLSTM2d(nn.Module):
    """Spatiotemporal LSTM (ST-LSTM) from PredRNN with optional depthwise-separable Conv.

    Compared to standard ConvLSTM, ST-LSTM maintains two memory cells:
      - C: temporal memory, updated via (x, h_{t-1}), same as ConvLSTM
      - M: spatial memory, updated via (x, m_{t-1}), propagated temporally within each stage
    The hidden state is derived from both memories:
      H_t = o_t * tanh(W_1x1([C_t, M_t]))

    When zigzag=false, M is carried across time within each stage independently.
    When zigzag=true, M flows across layers within each time step (l-1 -> l),
    and from the last layer at t-1 to the first layer at t (zigzag connection).

    Extras (PredRNN++-style training stability):
      - 5x5 dws on h (temporal) and on m (spatial path)
      - GroupNorm on IFG pre-activations and output path (bs=1 friendly)
      - Chrono init on forget gates (uses T_max_chrono_init from config per stage)
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 5,
                 dws_on_spatial_m: bool = True,
                 cell_update_dropout: float = 0.,
                 T_max_chrono_init: Optional[int] = None,
                 use_group_norm: bool = True):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim
        self.T_max_chrono_init = T_max_chrono_init
        self.use_group_norm = use_group_norm

        xh_dim = dim * 2
        t_ch = dim * 3

        # Depthwise-separable conv on h_{t-1} (temporal path, PredRNN 5x5)
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(
            in_channels=conv3x3_dws_dim,
            out_channels=conv3x3_dws_dim,
            kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2,
            groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        # Same receptive field on m_{t-1} in spatial path (separate from h)
        self.conv3x3_dws_m = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2,
            groups=dim) if (dws_conv and dws_on_spatial_m) else nn.Identity()
        self.conv_only_hidden = dws_conv_only_hidden
        self.dws_on_spatial_m = dws_on_spatial_m

        # Temporal memory path: cat(x, h_{t-1}) -> i, f, g  (3 * dim)
        self.conv1x1_temporal = nn.Conv2d(in_channels=xh_dim, out_channels=t_ch, kernel_size=1)
        # Spatial memory path: cat(x, m_{t-1}) -> i', f', g'  (3 * dim)
        self.conv1x1_spatial = nn.Conv2d(in_channels=xh_dim, out_channels=t_ch, kernel_size=1)

        # Output gate: cat(x, h_{t-1}, C_t, M_t) -> o  (dim)
        self.conv1x1_output = nn.Conv2d(in_channels=dim * 4, out_channels=dim, kernel_size=1)

        # Combine C and M for hidden state: cat(C_t, M_t) -> dim
        self.conv1x1_memory = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)

        g3 = _gn_num_groups(t_ch)
        g_dim = _gn_num_groups(dim)
        self.gn_temporal = nn.GroupNorm(g3, t_ch) if use_group_norm else nn.Identity()
        self.gn_spatial = nn.GroupNorm(g3, t_ch) if use_group_norm else nn.Identity()
        self.gn_output = nn.GroupNorm(g_dim, dim) if use_group_norm else nn.Identity()
        self.gn_memory = nn.GroupNorm(g_dim, dim) if use_group_norm else nn.Identity()

        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

        # Chrono: temporal + spatial forget gates
        if T_max_chrono_init is not None:
            if self.conv1x1_temporal.bias is not None:
                _chrono_ifg_bias(self.conv1x1_temporal.bias, dim, T_max_chrono_init, coupled=True)
            if self.conv1x1_spatial.bias is not None:
                _chrono_ifg_bias(self.conv1x1_spatial.bias, dim, T_max_chrono_init, coupled=True)

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None,
                m_previous: Optional[th.Tensor] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W)) temporal state, or None
        :param m_previous: (N C H W) spatial memory from zigzag/cross-layer routing, or None
        :return: (h_t, c_t, m_t), each (N C H W)
        """
        if h_and_c_previous is None:
            h_tm1 = th.zeros_like(x)
            c_tm1 = th.zeros_like(x)
        else:
            h_tm1, c_tm1 = h_and_c_previous

        m_tm1 = m_previous if m_previous is not None else th.zeros_like(x)

        # --- Temporal memory C update (same gating as ConvLSTM) ---
        if self.conv_only_hidden:
            h_tm1_conv = self.conv3x3_dws(h_tm1)
        else:
            h_tm1_conv = h_tm1

        xh = th.cat((x, h_tm1_conv), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)

        temporal_mix = self.conv1x1_temporal(xh)
        temporal_mix = self.gn_temporal(temporal_mix)
        t_i, t_f, t_g = th.tensor_split(temporal_mix, 3, dim=1)
        t_i = th.sigmoid(t_i)
        t_f = th.sigmoid(t_f)
        t_g = self.cell_update_dropout(th.tanh(t_g))
        c_t = t_f * c_tm1 + t_i * t_g

        # --- Spatial memory M update (5x5 dws on m when enabled) ---
        m_in = self.conv3x3_dws_m(m_tm1)
        xm = th.cat((x, m_in), dim=1)
        spatial_mix = self.conv1x1_spatial(xm)
        spatial_mix = self.gn_spatial(spatial_mix)
        s_i, s_f, s_g = th.tensor_split(spatial_mix, 3, dim=1)
        s_i = th.sigmoid(s_i)
        s_f = th.sigmoid(s_f)
        s_g = self.cell_update_dropout(th.tanh(s_g))
        m_t = s_f * m_tm1 + s_i * s_g

        # --- Output gate (combines all four sources) ---
        o_preact = self.conv1x1_output(th.cat((x, h_tm1, c_t, m_t), dim=1))
        o_preact = self.gn_output(o_preact)
        o_t = th.sigmoid(o_preact)

        # --- Hidden state from combined memories ---
        mem = self.conv1x1_memory(th.cat((c_t, m_t), dim=1))
        mem = self.gn_memory(mem)
        h_t = o_t * th.tanh(mem)

        return h_t, c_t, m_t
