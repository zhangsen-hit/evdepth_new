import math
from typing import Optional, Tuple

import torch as th
import torch.nn as nn


def _gn_num_groups(channels: int, preferred: int = 8) -> int:
    g = min(preferred, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


class RMSNorm2d(nn.Module):
    """Channel-wise RMSNorm for NCHW tensors (PyTorch nn.RMSNorm semantics).

    For each spatial location independently, normalizes across the channel dim
    by the RMS (no mean subtraction). Affine scale only (no bias). Stats are
    computed in float32 for fp16/bf16 stability, then cast back.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(th.ones(channels))
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        x32 = x.float()
        rms = x32.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        y = (x32 * rms).to(x.dtype)
        return y * self.weight.view(1, -1, 1, 1)


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

    # Component names for selective normalization (4 of them).
    NORM_COMPONENTS = ("temporal", "spatial", "output", "memory")

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 5,
                 dws_on_spatial_m: bool = True,
                 cell_update_dropout: float = 0.,
                 T_max_chrono_init: Optional[int] = None,
                 use_group_norm: bool = True,
                 norm_type: str = "gn",
                 norm_components: Optional[Tuple[str, ...]] = None):
        """
        norm_type: 'gn' (default), 'ln' (= GroupNorm with num_groups=1, equivalent
                   to LayerNorm over (C,H,W)), 'bn' (BatchNorm2d), or 'none'.
        norm_components: subset of {'temporal','spatial','output','memory'} that
                         get normalization. None = all four. [] = none.
        use_group_norm: legacy switch. False overrides everything to 'none'.
        """
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim
        self.T_max_chrono_init = T_max_chrono_init
        # Normalize the legacy + new flags into a single (norm_type, components) pair.
        if not use_group_norm:
            norm_type = "none"
            norm_components = ()
        if norm_components is None:
            norm_components = self.NORM_COMPONENTS
        norm_components = tuple(norm_components)
        for c in norm_components:
            if c not in self.NORM_COMPONENTS:
                raise ValueError(
                    f"unknown norm component {c!r}; expected subset of {self.NORM_COMPONENTS}"
                )
        if norm_type not in ("gn", "ln", "bn", "rms", "in", "none"):
            raise ValueError(
                f"norm_type must be 'gn'|'ln'|'bn'|'rms'|'in'|'none', got {norm_type!r}"
            )
        self.norm_type = norm_type
        self.norm_components = norm_components

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

        # Build per-component norm modules; Identity when disabled or component missing.
        self.gn_temporal = self._make_norm("temporal", t_ch)
        self.gn_spatial = self._make_norm("spatial", t_ch)
        self.gn_output = self._make_norm("output", dim)
        self.gn_memory = self._make_norm("memory", dim)

        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

        # Chrono: temporal + spatial forget gates.
        #
        # IMPORTANT: GN/IN with affine subtracts the mean over the channel/group,
        # which kills any per-channel additive bias on the conv output. So a
        # chrono offset written to conv1x1_*.bias is silently neutralized when
        # norm is non-Identity. Fix: write chrono to norm.bias (β, applied AFTER
        # normalization) when the norm has an affine bias; fall back to
        # conv.bias when norm is Identity (e.g., that component disabled).
        if T_max_chrono_init is not None:
            self._apply_chrono_init(self.conv1x1_temporal, self.gn_temporal, dim, T_max_chrono_init)
            self._apply_chrono_init(self.conv1x1_spatial, self.gn_spatial, dim, T_max_chrono_init)

    @staticmethod
    def _apply_chrono_init(conv: nn.Conv2d, norm: nn.Module, dim: int, T_max: int) -> None:
        """Write chrono forget-gate bias where it survives.

        Prefers norm's affine `bias` (β, post-norm); else falls back to conv's bias.
        """
        target = None
        if hasattr(norm, "bias") and isinstance(getattr(norm, "bias", None), nn.Parameter):
            target = norm.bias
        elif conv.bias is not None:
            target = conv.bias
        if target is not None:
            _chrono_ifg_bias(target, dim, T_max, coupled=True)

    def _make_norm(self, component: str, channels: int) -> nn.Module:
        """Build the norm module for a given component, honoring norm_type/components."""
        if self.norm_type == "none" or component not in self.norm_components:
            return nn.Identity()
        if self.norm_type == "gn":
            return nn.GroupNorm(_gn_num_groups(channels), channels)
        if self.norm_type == "ln":
            # GroupNorm with num_groups=1 == LayerNorm over (C,H,W) per sample.
            return nn.GroupNorm(1, channels)
        if self.norm_type == "bn":
            return nn.BatchNorm2d(channels)
        if self.norm_type == "rms":
            # Channel-wise RMSNorm per spatial location (PyTorch nn.RMSNorm convention).
            return RMSNorm2d(channels)
        if self.norm_type == "in":
            return nn.InstanceNorm2d(channels, affine=True)
        raise ValueError(self.norm_type)

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


class FusedDWSConvSTLSTM2d(nn.Module):
    """Fused-gate ST-LSTM (free fusion) — one DW pair + one PW for all 7 gates.

    Compared to DWSConvSTLSTM2d:
      - Keeps separate DW on h_{t-1} and m_{t-1} (cheap, semantic).
      - Fuses temporal/spatial/output gate PWs into ONE 1×1: 3·dim → 7·dim.
        Input is cat(x, dw(h), dw(m)) so every gate sees every signal — the
        "free fusion" variant; we let optimization decide if cross-talk helps.
      - One Norm on the 7·dim gates block (instead of 3 separate norms).
      - Memory mixer (cat(C,M) → dim) preserved as a small extra 1×1 + Norm.

    Drop-in for DWSConvSTLSTM2d: same forward signature, returns (h, c, m).

    Norm semantics:
      - norm_components ⊆ {"gates","memory"}. For backward compatibility,
        passing the original {temporal, spatial, output, memory} subset is
        also accepted: any of {temporal,spatial,output} → "gates",
        and "memory" → "memory".
    """

    NORM_COMPONENTS = ("gates", "memory")

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,  # noqa: ARG002, kept for API parity
                 dws_conv_kernel_size: int = 5,
                 dws_on_spatial_m: bool = True,
                 cell_update_dropout: float = 0.,
                 T_max_chrono_init: Optional[int] = None,
                 use_group_norm: bool = True,
                 norm_type: str = "gn",
                 norm_components: Optional[Tuple[str, ...]] = None):
        super().__init__()
        self.dim = dim
        self.T_max_chrono_init = T_max_chrono_init
        if not use_group_norm:
            norm_type = "none"
            norm_components = ()
        if norm_components is None:
            norm_components = self.NORM_COMPONENTS
        # Map original 4-component names down to {"gates","memory"} for the fused cell.
        comp_set = set(norm_components)
        gates_on = bool(comp_set & {"gates", "temporal", "spatial", "output"})
        memory_on = bool(comp_set & {"memory"})
        if norm_type not in ("gn", "ln", "bn", "rms", "in", "none"):
            raise ValueError(
                f"norm_type must be 'gn'|'ln'|'bn'|'rms'|'in'|'none', got {norm_type!r}"
            )
        self.norm_type = norm_type
        self._gates_on = gates_on
        self._memory_on = memory_on

        # Depthwise on h_{t-1} (temporal path receptive field).
        self.conv_dws_h = nn.Conv2d(
            dim, dim, kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2, groups=dim,
        ) if dws_conv else nn.Identity()
        # Depthwise on m_{t-1} (spatial path receptive field).
        self.conv_dws_m = nn.Conv2d(
            dim, dim, kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2, groups=dim,
        ) if (dws_conv and dws_on_spatial_m) else nn.Identity()

        # Fused gate PW: cat(x, dw(h), dw(m)) → (i, f, g, i_m, f_m, g_m, o).
        self.conv1x1_fused = nn.Conv2d(
            in_channels=dim * 3, out_channels=dim * 7, kernel_size=1,
        )
        # Memory mixer (linear bottleneck on combined C+M).
        self.conv1x1_memory = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim, kernel_size=1,
        )

        self.gn_gates = self._make_norm("gates", dim * 7)
        self.gn_memory = self._make_norm("memory", dim)

        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

        # Chrono init for both forget gates (temporal f at slice [d:2d] and
        # spatial f_m at [4d:5d]) of the fused 7·dim output.
        if T_max_chrono_init is not None:
            self._apply_fused_chrono(self.conv1x1_fused, self.gn_gates, dim, T_max_chrono_init)

    @staticmethod
    def _apply_fused_chrono(conv: nn.Conv2d, norm: nn.Module, dim: int, T_max: int) -> None:
        """Write chrono offsets to forget-gate slices of (norm or conv) bias.

        Layout: [i, f, g, i_m, f_m, g_m, o], each `dim` channels.
        """
        target = None
        if hasattr(norm, "bias") and isinstance(getattr(norm, "bias", None), nn.Parameter):
            target = norm.bias
        elif conv.bias is not None:
            target = conv.bias
        if target is None:
            return
        with th.no_grad():
            # Temporal forget (f) at [d:2d], coupled input (i) at [0:d].
            b_f_t = th.empty(dim).uniform_(math.log(1.5), math.log(float(T_max)))
            target[dim : 2 * dim].copy_(b_f_t)
            target[0 : dim].copy_(-b_f_t)
            # Spatial forget (f_m) at [4d:5d], coupled input (i_m) at [3d:4d].
            b_f_s = th.empty(dim).uniform_(math.log(1.5), math.log(float(T_max)))
            target[4 * dim : 5 * dim].copy_(b_f_s)
            target[3 * dim : 4 * dim].copy_(-b_f_s)

    def _make_norm(self, component: str, channels: int) -> nn.Module:
        on = self._gates_on if component == "gates" else self._memory_on
        if self.norm_type == "none" or not on:
            return nn.Identity()
        if self.norm_type == "gn":
            return nn.GroupNorm(_gn_num_groups(channels), channels)
        if self.norm_type == "ln":
            return nn.GroupNorm(1, channels)
        if self.norm_type == "bn":
            return nn.BatchNorm2d(channels)
        if self.norm_type == "rms":
            return RMSNorm2d(channels)
        if self.norm_type == "in":
            return nn.InstanceNorm2d(channels, affine=True)
        raise ValueError(self.norm_type)

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None,
                m_previous: Optional[th.Tensor] = None
                ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if h_and_c_previous is None:
            h_tm1 = th.zeros_like(x)
            c_tm1 = th.zeros_like(x)
        else:
            h_tm1, c_tm1 = h_and_c_previous
        m_tm1 = m_previous if m_previous is not None else th.zeros_like(x)

        h_dw = self.conv_dws_h(h_tm1)
        m_dw = self.conv_dws_m(m_tm1)

        gates = self.conv1x1_fused(th.cat((x, h_dw, m_dw), dim=1))
        gates = self.gn_gates(gates)
        # Split into 7 chunks of `dim` channels each: i, f, g, i_m, f_m, g_m, o.
        i, f, g, i_m, f_m, g_m, o = th.tensor_split(gates, 7, dim=1)

        c_t = th.sigmoid(f) * c_tm1 + th.sigmoid(i) * self.cell_update_dropout(th.tanh(g))
        m_t = th.sigmoid(f_m) * m_tm1 + th.sigmoid(i_m) * self.cell_update_dropout(th.tanh(g_m))

        mem = self.conv1x1_memory(th.cat((c_t, m_t), dim=1))
        mem = self.gn_memory(mem)
        h_t = th.sigmoid(o) * th.tanh(mem)

        return h_t, c_t, m_t
