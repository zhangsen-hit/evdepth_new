"""
LiOSAM odometry2 depth_dataset 支持。

数据格式：
- 目录下 index.txt：每行 frame_id, timestamp_sec, filename
- 目录下 *.npz：每个 npz 含事件张量 (2, 260, 346) 与深度图
- 序列：时间上连续的 sequence_length 帧为一组，相邻帧间隔约 5ms，允许 ±3ms 偏差
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import DataType, LoaderDataDictGenX


def center_crop_tensor_2d(
    tensor: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """
    对末尾两维空间维做居中裁剪：(C,H,W) 或其它 (*, H, W)。
    若 spatial 尺寸小于目标则不抛错，仅按可用范围截取。
    """
    h = int(tensor.shape[-2])
    w = int(tensor.shape[-1])
    if h >= out_h:
        top = max(0, (h - out_h) // 2)
        bot = top + out_h
    else:
        top, bot = 0, h
    if w >= out_w:
        left = max(0, (w - out_w) // 2)
        right = left + out_w
    else:
        left, right = 0, w
    return tensor[..., top:bot, left:right]


def normalize_events_nonzero_channels(ev_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    ev_t: (C, H, W)。逐通道在非零位置上减均值除以标准差；零保持为零。
    """
    assert ev_t.dim() == 3
    out = ev_t.clone()
    for c in range(out.shape[0]):
        plane = out[c]
        nz = plane != 0
        if nz.any():
            vals = plane[nz]
            mu = vals.mean()
            sig = vals.std(unbiased=False).clamp(min=eps)
            plane[nz] = (plane[nz] - mu) / sig
    return out


# 默认 npz 中事件与深度的 key（可配置）
# 常见结构: input=(2,260,346) 事件, label=(260,346) 深度（含 inf 为无效）
DEFAULT_EV_KEY = "input"
DEFAULT_DEPTH_KEY = "label"
DEFAULT_DEPTH_MASK_KEY = None  # 无则从 label 的 isfinite 与范围生成


def _load_depth_and_mask_from_npz(
    data: Any,
    depth_key: str,
    depth_mask_key: Optional[str],
    min_depth: float,
    max_depth: float,
    convert_to_log: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 npz 的 data 中加载深度与有效 mask。支持 label 中含 inf/nan（对应像素 mask 为 False）。
    返回 (depth_tensor (1,H,W), mask_tensor (1,H,W))。
    """
    depth_np = np.asarray(data[depth_key], dtype=np.float32)
    finite = np.isfinite(depth_np)
    depth_np_safe = np.where(finite, np.clip(depth_np, min_depth, max_depth), max_depth)
    depth_t = torch.from_numpy(depth_np_safe).float()
    if depth_t.dim() == 2:
        depth_t = depth_t.unsqueeze(0)  # (1, H, W)
    if convert_to_log:
        depth_t = torch.log(depth_t)

    if depth_mask_key and depth_mask_key in data:
        mask = data[depth_mask_key]
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.uint8)).bool()
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
    else:
        depth_linear = depth_t.exp() if convert_to_log else depth_t
        mask_t = (
            torch.from_numpy(finite).bool()
            & (depth_linear > min_depth)
            & (depth_linear < max_depth)
        )
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
    return depth_t, mask_t


def load_liosam_index(path: Path) -> List[Tuple[int, float, str]]:
    """
    解析 index.txt，返回 [(frame_id, timestamp_sec, filename), ...]，按 frame_id 排序。
    """
    index_file = path / "index.txt"
    if not index_file.exists():
        raise FileNotFoundError(f"index.txt not found: {index_file}")
    entries = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            frame_id = int(parts[0])
            timestamp = float(parts[1])
            filename = parts[2]
            entries.append((frame_id, timestamp, filename))
    entries.sort(key=lambda x: x[0])
    return entries


def find_contiguous_runs(
    entries: List[Tuple[int, float, str]],
    min_dt_sec: float = 0.002,
    max_dt_sec: float = 0.008,
) -> List[List[int]]:
    """
    根据时间戳将 entries 划分为连续片段。相邻帧 dt 需在 [min_dt_sec, max_dt_sec] 内。
    返回每个连续片段的索引列表（每个元素是 entries 的下标列表）。
    """
    if not entries:
        return []
    runs = []
    current_run = [0]
    for i in range(1, len(entries)):
        dt = entries[i][1] - entries[i - 1][1]
        if min_dt_sec <= dt <= max_dt_sec:
            current_run.append(i)
        else:
            if len(current_run) >= 1:
                runs.append(current_run)
            current_run = [i]
    if current_run:
        runs.append(current_run)
    return runs


def build_sequence_windows(
    runs: List[List[int]],
    sequence_length: int,
) -> List[Tuple[int, int, List[int]]]:
    """
    将每个 run 切分为长度为 sequence_length 的窗口（不重叠）。
    返回 [(run_idx, start_in_run, end_in_run), ...]，其中每个 (start,end) 对应
    run 内 indices[start:end]，长度为 sequence_length。
    若某 run 长度不足 sequence_length，则跳过该 run。
    """
    windows = []
    for run_idx, run in enumerate(runs):
        n = len(run)
        for start in range(0, n - sequence_length + 1, sequence_length):
            end = start + sequence_length
            windows.append((run_idx, start, end, run[start:end]))
    return windows


class LiosamSequenceForIter(Dataset):
    """
    单个 200 帧（可配置）序列的流式访问。从 npz 按帧加载事件与深度。
    """

    def __init__(
        self,
        path: Path,
        frame_indices: List[int],
        entries: List[Tuple[int, float, str]],
        sequence_length: int,
        ev_key: str = DEFAULT_EV_KEY,
        depth_key: str = DEFAULT_DEPTH_KEY,
        depth_mask_key: Optional[str] = DEFAULT_DEPTH_MASK_KEY,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        convert_depth_to_log: bool = True,
        center_crop_hw: Optional[Tuple[int, int]] = None,
        normalize_events_nonzero: bool = False,
    ):
        self.path = path
        self.frame_indices = frame_indices  # indices into entries for this sequence
        self.entries = entries
        self.seq_len = sequence_length
        self.ev_key = ev_key
        self.depth_key = depth_key
        self.depth_mask_key = depth_mask_key
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.convert_depth_to_log = convert_depth_to_log
        self.center_crop_hw = (
            tuple(int(x) for x in center_crop_hw) if center_crop_hw is not None else None
        )
        self.normalize_events_nonzero = bool(normalize_events_nonzero)
        self._padding_representation = None
        assert len(frame_indices) == sequence_length

    @property
    def padding_representation(self) -> torch.Tensor:
        if self._padding_representation is None:
            fn = self.path / self.entries[self.frame_indices[0]][2]
            data = np.load(str(fn), allow_pickle=True)
            ev = data[self.ev_key]
            t = torch.zeros_like(torch.from_numpy(ev).float())
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if self.center_crop_hw is not None:
                ch, cw = self.center_crop_hw
                t = center_crop_tensor_2d(t, ch, cw)
            self._padding_representation = t
        return self._padding_representation

    def get_fully_padded_sample(self) -> LoaderDataDictGenX:
        is_first_sample = False
        is_padded_mask = [True] * self.seq_len
        ev_repr = [self.padding_representation] * self.seq_len
        labels = [None] * self.seq_len
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        # 深度占位
        d = self.padding_representation
        if d.dim() == 2:
            depth_placeholder = torch.zeros(1, d.shape[0], d.shape[1], dtype=torch.float32)
        else:
            depth_placeholder = torch.zeros(1, d.shape[1], d.shape[2], dtype=torch.float32)
        mask_placeholder = torch.zeros_like(depth_placeholder, dtype=torch.bool)
        return {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
            DataType.DEPTH: [depth_placeholder] * self.seq_len,
            DataType.DEPTH_MASK: [mask_placeholder] * self.seq_len,
        }

    def __len__(self) -> int:
        return 1  # 每个 LiosamSequenceForIter 只代表一个 200 帧样本

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        assert index == 0
        ev_repr = []
        depths = []
        masks = []
        for idx in self.frame_indices:
            _, _, filename = self.entries[idx]
            fn = self.path / filename
            data = np.load(str(fn), allow_pickle=True)
            ev = data[self.ev_key]
            ev_t = torch.from_numpy(ev).float()
            if ev_t.dim() == 2:
                ev_t = ev_t.unsqueeze(0)
            if self.center_crop_hw is not None:
                ch, cw = self.center_crop_hw
                ev_t = center_crop_tensor_2d(ev_t, ch, cw)
            if self.normalize_events_nonzero:
                ev_t = normalize_events_nonzero_channels(ev_t)
            ev_repr.append(ev_t)

            depth, mask_t = _load_depth_and_mask_from_npz(
                data, self.depth_key, self.depth_mask_key,
                self.min_depth, self.max_depth, self.convert_depth_to_log,
            )
            if self.center_crop_hw is not None:
                ch, cw = self.center_crop_hw
                depth = center_crop_tensor_2d(depth, ch, cw)
                mask_t = center_crop_tensor_2d(mask_t, ch, cw)
            depths.append(depth)
            masks.append(mask_t)

        sparse_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=[None] * self.seq_len
        )
        return {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: True,
            DataType.IS_PADDED_MASK: [False] * self.seq_len,
            DataType.DEPTH: depths,
            DataType.DEPTH_MASK: masks,
        }


class LiosamSequenceForRandomAccess:
    """
    随机访问的 200 帧序列。与 LiosamSequenceForIter 相同数据，接口兼容 SequenceForRandomAccess。
    """

    def __init__(
        self,
        path: Path,
        frame_indices: List[int],
        entries: List[Tuple[int, float, str]],
        sequence_length: int,
        ev_key: str = DEFAULT_EV_KEY,
        depth_key: str = DEFAULT_DEPTH_KEY,
        depth_mask_key: Optional[str] = DEFAULT_DEPTH_MASK_KEY,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        convert_depth_to_log: bool = True,
        center_crop_hw: Optional[Tuple[int, int]] = None,
        normalize_events_nonzero: bool = False,
    ):
        self.path = path
        self.frame_indices = frame_indices
        self.entries = entries
        self.seq_len = sequence_length
        self.ev_key = ev_key
        self.depth_key = depth_key
        self.depth_mask_key = depth_mask_key
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.convert_depth_to_log = convert_depth_to_log
        self.center_crop_hw = (
            tuple(int(x) for x in center_crop_hw) if center_crop_hw is not None else None
        )
        self.normalize_events_nonzero = bool(normalize_events_nonzero)
        self.length = 1
        self._only_load_labels = False

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        assert index == 0
        ev_repr = []
        depths = []
        masks = []
        for idx in self.frame_indices:
            _, _, filename = self.entries[idx]
            fn = self.path / filename
            data = np.load(str(fn), allow_pickle=True)
            ev = data[self.ev_key]
            ev_t = torch.from_numpy(ev).float()
            if ev_t.dim() == 2:
                ev_t = ev_t.unsqueeze(0)
            if self.center_crop_hw is not None:
                ch, cw = self.center_crop_hw
                ev_t = center_crop_tensor_2d(ev_t, ch, cw)
            if self.normalize_events_nonzero:
                ev_t = normalize_events_nonzero_channels(ev_t)
            ev_repr.append(ev_t)

            depth, mask_t = _load_depth_and_mask_from_npz(
                data, self.depth_key, self.depth_mask_key,
                self.min_depth, self.max_depth, self.convert_depth_to_log,
            )
            if self.center_crop_hw is not None:
                ch, cw = self.center_crop_hw
                depth = center_crop_tensor_2d(depth, ch, cw)
                mask_t = center_crop_tensor_2d(mask_t, ch, cw)
            depths.append(depth)
            masks.append(mask_t)

        sparse_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=[None] * self.seq_len
        )
        return {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: True,
            DataType.IS_PADDED_MASK: [False] * self.seq_len,
            DataType.DEPTH: depths,
            DataType.DEPTH_MASK: masks,
        }

    def only_load_labels(self):
        self._only_load_labels = True

    def load_everything(self):
        self._only_load_labels = False

    def is_only_loading_labels(self) -> bool:
        return self._only_load_labels


def _extract_common_config(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """从 dataset_config 中提取构建序列所需的公共参数。"""
    depth_range = dataset_config.get("depth_range", {})
    chw = dataset_config.get("center_crop_hw", None)
    if chw is not None:
        chw = tuple(int(x) for x in chw)
    return dict(
        sequence_length=dataset_config["sequence_length"],
        min_dt=dataset_config.get("frame_interval_sec", 0.005) - dataset_config.get("max_interval_deviation_sec", 0.003),
        max_dt=dataset_config.get("frame_interval_sec", 0.005) + dataset_config.get("max_interval_deviation_sec", 0.003),
        ev_key=dataset_config.get("ev_key", DEFAULT_EV_KEY),
        depth_key=dataset_config.get("depth_key", DEFAULT_DEPTH_KEY),
        depth_mask_key=dataset_config.get("depth_mask_key", DEFAULT_DEPTH_MASK_KEY),
        min_depth=float(depth_range.get("min", 0.1)),
        max_depth=float(depth_range.get("max", 100.0)),
        center_crop_hw=chw,
        normalize_events_nonzero=bool(dataset_config.get("normalize_events_nonzero", False)),
    )


# (scene_path, entries, window_indices) — 每个元素描述一个来自某场景的窗口
SceneWindow = Tuple[Path, List[Tuple[int, float, str]], List[int]]


def _build_windows_for_scene(
    scene_path: Path,
    sequence_length: int,
    min_dt: float,
    max_dt: float,
) -> List[SceneWindow]:
    """
    对单个场景目录构建所有 sequence windows。
    返回 [(scene_path, entries, window_indices), ...] 列表。
    """
    entries = load_liosam_index(scene_path)
    if not entries:
        return []
    runs = find_contiguous_runs(entries, min_dt_sec=min_dt, max_dt_sec=max_dt)
    scene_windows: List[SceneWindow] = []
    for run in runs:
        n = len(run)
        for start in range(0, n - sequence_length + 1, sequence_length):
            end = start + sequence_length
            scene_windows.append((scene_path, entries, run[start:end]))
    return scene_windows


def _make_stream_list(
    windows: List[SceneWindow],
    cfg: Dict[str, Any],
) -> List[LiosamSequenceForIter]:
    return [
        LiosamSequenceForIter(
            path=scene_path,
            frame_indices=wnd,
            entries=entries,
            sequence_length=cfg["sequence_length"],
            ev_key=cfg["ev_key"],
            depth_key=cfg["depth_key"],
            depth_mask_key=cfg["depth_mask_key"],
            min_depth=cfg["min_depth"],
            max_depth=cfg["max_depth"],
            convert_depth_to_log=True,
            center_crop_hw=cfg.get("center_crop_hw"),
            normalize_events_nonzero=cfg.get("normalize_events_nonzero", False),
        )
        for scene_path, entries, wnd in windows
    ]


def _make_rnd_list(
    windows: List[SceneWindow],
    cfg: Dict[str, Any],
) -> List[LiosamSequenceForRandomAccess]:
    return [
        LiosamSequenceForRandomAccess(
            path=scene_path,
            frame_indices=wnd,
            entries=entries,
            sequence_length=cfg["sequence_length"],
            ev_key=cfg["ev_key"],
            depth_key=cfg["depth_key"],
            depth_mask_key=cfg["depth_mask_key"],
            min_depth=cfg["min_depth"],
            max_depth=cfg["max_depth"],
            convert_depth_to_log=True,
            center_crop_hw=cfg.get("center_crop_hw"),
            normalize_events_nonzero=cfg.get("normalize_events_nonzero", False),
        )
        for scene_path, entries, wnd in windows
    ]


def build_liosam_sequences(
    path: Path,
    dataset_config: Dict[str, Any],
    train_ratio: float = 0.9,
) -> Tuple[
    List[LiosamSequenceForIter],
    List[LiosamSequenceForIter],
    List[LiosamSequenceForRandomAccess],
    List[LiosamSequenceForRandomAccess],
]:
    """
    构建 LiOSAM 序列窗口，返回 (train_stream, val_stream, train_rnd, val_rnd)。

    支持两种模式：
    1) 多场景模式：dataset_config 中配置 train_scenes / val_scenes，
       path 下包含以场景名命名的子目录（如 00, 01, ...），每个子目录含 index.txt + npz。
    2) 单场景回退：path 下直接含 index.txt + npz，按 train_ratio 比例划分。
    """
    cfg = _extract_common_config(dataset_config)
    train_scenes = dataset_config.get("train_scenes", None)
    val_scenes = dataset_config.get("val_scenes", None)

    if train_scenes is not None and val_scenes is not None:
        # ---------- 多场景模式 ----------
        train_windows: List[SceneWindow] = []
        val_windows: List[SceneWindow] = []

        for scene_name in train_scenes:
            scene_path = path / str(scene_name)
            assert scene_path.is_dir(), f"Train scene directory not found: {scene_path}"
            sw = _build_windows_for_scene(scene_path, cfg["sequence_length"], cfg["min_dt"], cfg["max_dt"])
            print(f'[liosam] Scene {scene_name}: {len(sw)} windows -> TRAIN')
            train_windows.extend(sw)

        for scene_name in val_scenes:
            scene_path = path / str(scene_name)
            assert scene_path.is_dir(), f"Val scene directory not found: {scene_path}"
            sw = _build_windows_for_scene(scene_path, cfg["sequence_length"], cfg["min_dt"], cfg["max_dt"])
            print(f'[liosam] Scene {scene_name}: {len(sw)} windows -> VAL')
            val_windows.extend(sw)

        print(f'[liosam] Total: train={len(train_windows)} windows, val={len(val_windows)} windows')

    else:
        # ---------- 单场景回退（向后兼容） ----------
        all_windows = _build_windows_for_scene(path, cfg["sequence_length"], cfg["min_dt"], cfg["max_dt"])
        if not all_windows:
            return [], [], [], []
        n = len(all_windows)
        n_train = max(1, int(n * train_ratio))
        train_windows = all_windows[:n_train]
        val_windows = all_windows[n_train:]

    if not train_windows and not val_windows:
        return [], [], [], []

    train_stream = _make_stream_list(train_windows, cfg)
    val_stream = _make_stream_list(val_windows, cfg)
    train_rnd = _make_rnd_list(train_windows, cfg)
    val_rnd = _make_rnd_list(val_windows, cfg)
    return train_stream, val_stream, train_rnd, val_rnd


class LiosamSequenceDataset:
    """
    包装单个 LiosamSequenceForRandomAccess，使 CustomConcatDataset 可拼接多个序列。
    """

    def __init__(self, sequence: LiosamSequenceForRandomAccess):
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        return self.sequence[index]

    def only_load_labels(self):
        self.sequence.only_load_labels()

    def load_everything(self):
        self.sequence.load_everything()
