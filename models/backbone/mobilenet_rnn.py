"""
MobileNetV2 + RNN Backbone for Depth Estimation
Simplified version with only MobileNetV2 blocks
"""
from typing import Dict, Optional, Tuple, Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.backbone.rnn import (
    DWSConvLSTM2d,
    DWSConvSTLSTM2d,
    StandardConvLSTM2d,
    _gn_num_groups,
)
from models.backbone.base import BaseDetector


class MobileNetV2Block(nn.Module):
    """
    MobileNetV2 Inverted Residual Block
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(MobileNetV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetRNNStage(nn.Module):
    """
    Single stage of MobileNet + RNN
    """
    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: Dict[str, Any],
                 stage_idx: int = 0):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        lstm_cfg = stage_cfg.get('lstm', {})

        self.spatial_downsample_factor = spatial_downsample_factor
        self.stage_idx = stage_idx

        # Downsampling layer
        if stage_idx == 0:
            # First stage: patch embedding
            kernel_size = spatial_downsample_factor
            stride = spatial_downsample_factor
            padding = 0
        else:
            # Subsequent stages: strided conv
            kernel_size = 2
            stride = 2
            padding = 0
        
        self.downsample = nn.Sequential(
            nn.Conv2d(dim_in, stage_dim, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(stage_dim),
            nn.ReLU6(inplace=True)
        )

        # MobileNetV2 blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(MobileNetV2Block(stage_dim, stage_dim, 1, 6))  # expand_ratio=6
        self.blocks = nn.Sequential(*blocks)

        # Recurrent cell: ConvLSTM or ST-LSTM
        cell_type = lstm_cfg.get('cell_type', 'convlstm')
        cell_kwargs = dict(
            dim=stage_dim,
            dws_conv=lstm_cfg.get('dws_conv', False),
            dws_conv_only_hidden=lstm_cfg.get('dws_conv_only_hidden', True),
            dws_conv_kernel_size=lstm_cfg.get('dws_conv_kernel_size', 3),
            cell_update_dropout=lstm_cfg.get('drop_cell_update', 0),
            T_max_chrono_init=T_max_chrono_init,
        )
        if cell_type == 'stlstm':
            cell_kwargs['dws_on_spatial_m'] = lstm_cfg.get('dws_on_spatial_m', True)
            cell_kwargs['use_group_norm'] = lstm_cfg.get('use_group_norm', True)
            self.lstm = DWSConvSTLSTM2d(**cell_kwargs)
        elif cell_type in ('convlstm', 'dws_convlstm'):
            self.lstm = DWSConvLSTM2d(**cell_kwargs)
        elif cell_type == 'stand_convlstm':
            stand_kwargs = dict(
                dim=stage_dim,
                cell_update_dropout=lstm_cfg.get('drop_cell_update', 0),
                T_max_chrono_init=T_max_chrono_init,
            )
            self.lstm = StandardConvLSTM2d(**stand_kwargs)
        else:
            raise ValueError(
                f"Unknown cell_type: {cell_type!r}; expected "
                f"'stlstm','convlstm','dws_convlstm','stand_convlstm'"
            )

        # Mask token for token masking (if enabled)
        self.mask_token = nn.Parameter(
            th.zeros(1, 1, 1, stage_dim),
            requires_grad=True
        ) if enable_token_masking else None
        
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)

    def forward(self,
                x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                m_input: Optional[th.Tensor] = None,
                token_mask: Optional[th.Tensor] = None,
                expected_hw: Optional[tuple] = None) \
            -> Tuple[FeatureMap, LstmState]:
        """
        Forward pass through stage
        
        Args:
            x: Input tensor (B, C, H, W)
            h_and_c_previous: Previous temporal state (h, c) for ConvLSTM/ST-LSTM
            m_input: Spatial memory M from zigzag/cross-layer routing (ST-LSTM only)
            token_mask: Token mask for masking
            expected_hw: Expected output spatial dimensions
        
        Returns:
            Feature map and LSTM state
        """
        # Automatic padding to ensure proper downsampling
        h, w = x.shape[2:]
        factor = self.spatial_downsample_factor
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = th.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        
        # Downsample
        x = self.downsample(x)
        
        # MobileNet blocks
        x = self.blocks(x)
        
        # Apply token masking if provided
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token
        
        # Recurrent cell
        if isinstance(self.lstm, DWSConvSTLSTM2d):
            h_t, c_t, m_t = self.lstm(x, h_and_c_previous, m_input)
            lstm_state = (h_t, c_t, m_t)
        else:
            lstm_state = self.lstm(x, h_and_c_previous)
        x = lstm_state[0]
        
        # Verify output dimensions
        if expected_hw is not None:
            assert x.shape[2] == expected_hw[0] and x.shape[3] == expected_hw[1], \
                f"Stage {self.stage_idx+1} output size {x.shape[2:]} != expected {expected_hw}"
        
        return x, lstm_state


class MobileNetRNN(BaseDetector):
    """
    MobileNetV2 + RNN backbone for depth estimation
    """
    def __init__(self, mdl_config: Dict[str, Any]):
        super().__init__()

        # Extract config
        in_channels = mdl_config['input_channels']
        embed_dim = mdl_config['embed_dim']
        dim_multiplier_per_stage = tuple(mdl_config['dim_multiplier'])
        num_blocks_per_stage = tuple(mdl_config['num_blocks'])
        T_max_chrono_init_per_stage = tuple(mdl_config['T_max_chrono_init'])
        enable_masking = mdl_config.get('enable_masking', False)

        stage_cfg = mdl_config.get('stage', {})
        cell_type = stage_cfg.get('lstm', {}).get('cell_type', 'convlstm')

        print("\n" + "="*60)
        print("Initializing MobileNetV2 + RNN Backbone")
        print("="*60)
        print(f"  Input channels: {in_channels}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Stage dimensions: {[embed_dim * x for x in dim_multiplier_per_stage]}")
        print(f"  Blocks per stage: {num_blocks_per_stage}")
        print(f"  Recurrent cell type: {cell_type}")
        if cell_type == 'stlstm':
            z = stage_cfg.get('lstm', {}).get('zigzag', False)
            print(f"  ST-LSTM zigzag M routing: {z}")
        print(f"  Token masking: {enable_masking}")
        print("="*60 + "\n")

        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4, "MobileNetRNN requires 4 stages"

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        # Compile if requested
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.get('enable', False) if isinstance(compile_cfg, dict) else False
            if compile_mdl and th_compile is not None:
                compile_args = compile_cfg.get('args', {}) if isinstance(compile_cfg, dict) else {}
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')

        # Build stages
        input_dim = in_channels
        stem_cfg = mdl_config.get('stem', {})
        patch_size = stem_cfg.get('patch_size', 4) if isinstance(stem_cfg, dict) else 4
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.use_zigzag = (cell_type == 'stlstm' and
                           stage_cfg.get('lstm', {}).get('zigzag', False))

        self.stages = nn.ModuleList()
        self.strides = []
        
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            
            stage = MobileNetRNNStage(
                dim_in=input_dim,
                stage_dim=stage_dim,
                spatial_downsample_factor=spatial_downsample_factor,
                num_blocks=num_blocks,
                enable_token_masking=enable_masking_in_stage,
                T_max_chrono_init=T_max_chrono_init_stage,
                stage_cfg=mdl_config.get('stage', {}),
                stage_idx=stage_idx
            )
            
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)
            self.stages.append(stage)
            input_dim = stage_dim

        # Zigzag M adapters: AvgPool + 1x1 + GN + SiLU (more stable than strided 2x2)
        if self.use_zigzag:
            d0 = self.stage_dims[0]
            d_last = self.stage_dims[-1]
            g0 = _gn_num_groups(d0)
            # Forward: stage l -> l+1 (2x downsample)
            self.m_adapters_forward = nn.ModuleList()
            for i in range(num_stages - 1):
                di, d_next = self.stage_dims[i], self.stage_dims[i + 1]
                g = _gn_num_groups(d_next)
                self.m_adapters_forward.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(di, d_next, kernel_size=1, bias=False),
                        nn.GroupNorm(g, d_next),
                        nn.SiLU(inplace=True),
                    )
                )
            # Zigzag: last -> first resolution (1x1 project, then upsample in forward, then refine)
            self.m_adapter_zigzag = nn.Sequential(
                nn.Conv2d(d_last, d0, kernel_size=1, bias=False),
                nn.GroupNorm(g0, d0),
                nn.SiLU(inplace=True),
            )
            self.m_adapter_zigzag_refine = nn.Sequential(
                nn.Conv2d(d0, d0, kernel_size=5, padding=2, groups=d0, bias=False),
                nn.GroupNorm(g0, d0),
                nn.SiLU(inplace=True),
                nn.Conv2d(d0, d0, kernel_size=1, bias=False),
            )
            print("  Zigzag M adapters: enabled (pool+1x1+GN+SiLU, zigzag 5x5 DW refine)")

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get dimensions of specified stages"""
        # Convert 1-based stage numbers to 0-based indices
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, f"Invalid stage index: {min(stage_indices)}"
        assert max(stage_indices) < len(self.stage_dims), \
            f"Stage index {max(stage_indices)} out of range [0, {len(self.stage_dims)-1}]"
        return tuple(self.stage_dims[i] for i in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get strides of specified stages"""
        # Convert 1-based stage numbers to 0-based indices
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, f"Invalid stage index: {min(stage_indices)}"
        assert max(stage_indices) < len(self.strides), \
            f"Stage index {max(stage_indices)} out of range [0, {len(self.strides)-1}]"
        return tuple(self.strides[i] for i in stage_indices)

    def forward(self,
                x: th.Tensor,
                prev_states: Optional[LstmStates] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        """
        Forward pass through MobileNet + RNN backbone
        
        Args:
            x: Input tensor (B, C, H, W)
            prev_states: Previous LSTM states for all stages
            token_mask: Token mask for first stage
        
        Returns:
            features: Dict of features from each stage
            states: Tuple of LSTM states
        """
        # Automatic padding to ensure dimensions are divisible by 32
        h, w = x.shape[2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = th.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        
        if prev_states is None:
            prev_states = [None] * len(self.stages)
        
        features = {}
        states = []
        
        # Expected output sizes for each stage
        expected_hw_list = [
            (x.size(2)//4, x.size(3)//4),    # Stage 1: /4
            (x.size(2)//8, x.size(3)//8),    # Stage 2: /8
            (x.size(2)//16, x.size(3)//16),  # Stage 3: /16
            (x.size(2)//32, x.size(3)//32),  # Stage 4: /32
        ]
        
        # --- Zigzag: extract M from last stage at previous time step ---
        m_current = None
        if self.use_zigzag and prev_states[-1] is not None and len(prev_states[-1]) >= 3:
            m_last = prev_states[-1][2]
            m_current = self.m_adapter_zigzag(m_last)
            m_current = F.interpolate(
                m_current, size=expected_hw_list[0],
                mode='bilinear', align_corners=False)
            m_current = self.m_adapter_zigzag_refine(m_current)
        
        for stage_idx, (stage, prev_state) in enumerate(zip(self.stages, prev_states)):
            expected_hw = expected_hw_list[stage_idx]
            
            # Separate temporal state (h, c) from spatial memory M
            h_c_prev = None
            m_for_stage = None
            
            if prev_state is not None and len(prev_state) >= 3:
                h_c_prev = (prev_state[0], prev_state[1])
                if self.use_zigzag:
                    m_for_stage = m_current
                else:
                    m_for_stage = prev_state[2]
            elif prev_state is not None:
                h_c_prev = prev_state
                m_for_stage = m_current if self.use_zigzag else None
            else:
                m_for_stage = m_current if self.use_zigzag else None
            
            x, state = stage(
                x, h_c_prev, m_for_stage,
                token_mask=token_mask, expected_hw=expected_hw)
            
            # Route M output to next stage via adapter
            if self.use_zigzag and len(state) >= 3:
                m_output = state[2]
                if stage_idx < len(self.stages) - 1:
                    m_current = self.m_adapters_forward[stage_idx](m_output)
            
            features[stage_idx + 1] = x
            states.append(state)
        
        return features, tuple(states)

