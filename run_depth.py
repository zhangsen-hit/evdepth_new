#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续推理脚本：在测试场景（dataset/22, dataset/23）上按时间顺序逐帧前向，
保存：
  - 每帧深度预测 (.npy, 米, 256x344)
  - 每帧 4 张可视化 PNG（事件 / 带 mask 预测 / 全预测 / GT）
  - 整段拼接的 200fps 视频（每一帧把 4 张图按原始分辨率左右横排）

数据预处理与 LiosamSequenceForRandomAccess 完全一致：center_crop (256,344)
+ normalize_events_nonzero。隐藏状态在同一连续片段内向后传递；遇到时间不连续
（dt 超出 [0.002, 0.008] 秒）或场景切换时清零。
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import shutil
import subprocess
import numpy as np
import torch
import yaml
from einops import rearrange, reduce
from PIL import Image

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.genx_utils.liosam_sequence import (
    DEFAULT_DEPTH_KEY,
    DEFAULT_EV_KEY,
    DEFAULT_DEPTH_MASK_KEY,
    center_crop_tensor_2d,
    load_liosam_index,
    normalize_events_nonzero_channels,
)
from modules.depth_estimation import Module as DepthModule, _finest_depth_pred


# --------------------------------------------------------------------------- #
# 工具：可视化
# --------------------------------------------------------------------------- #
def _depth_to_colormap(
    depth_m: np.ndarray,
    vmin: float,
    vmax: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """与 callbacks/depth_viz.py 中 _depth_to_colormap 完全一致：jet + log + 红近蓝远 + 黑无效。"""
    import matplotlib.cm as cm

    values = np.asarray(depth_m, dtype=np.float32)
    h, w = values.shape
    colormap = cm.get_cmap("jet")
    out = np.zeros((h, w, 3), dtype=np.uint8)
    eps = 1e-6

    valid = np.isfinite(values) & (values > 0)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    if not valid.any():
        return out

    vmin_eff = max(vmin, eps)
    vmax_eff = max(vmax, vmin_eff + eps)
    log_vmin = math.log(vmin_eff)
    log_vmax = math.log(vmax_eff)
    denom = max(log_vmax - log_vmin, 1e-12)

    log_values_valid = np.log(np.clip(values[valid], vmin_eff, None))
    norm_valid = np.clip((log_values_valid - log_vmin) / denom, 0.0, 1.0)
    norm_valid_rev = 1.0 - norm_valid
    out[valid] = (colormap(norm_valid_rev)[:, :3] * 255).astype(np.uint8)
    return out


def _ev_repr_to_img_white(x: np.ndarray, use_last_k_bins: int = 2) -> np.ndarray:
    """事件张量可视化，红=正极性、蓝=负极性，无事件处为白色背景。"""
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev = rearrange(x, "(posneg C) H W -> posneg C H W", posneg=2)
    num_bins = ev.shape[1]
    sel = slice(max(0, num_bins - use_last_k_bins), num_bins)
    img_neg = np.asarray(reduce(ev[0, sel], "C H W -> H W", "sum"), dtype=np.float32)
    img_pos = np.asarray(reduce(ev[1, sel], "C H W -> H W", "sum"), dtype=np.float32)

    def _norm_log_minmax(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        v = np.log1p(np.abs(v)).astype(np.float32)
        v_min = float(v.min())
        v_max = float(v.max())
        return np.clip((v - v_min) / (v_max - v_min + eps), 0.0, 1.0).astype(np.float32)

    r = _norm_log_minmax(img_pos)
    b = _norm_log_minmax(img_neg)

    img = np.ones((ht, wd, 3), dtype=np.float32)
    img[..., 0] -= b
    img[..., 1] -= np.maximum(r, b)
    img[..., 2] -= r
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 数据预处理（与 LiosamSequenceForRandomAccess 一致）
# --------------------------------------------------------------------------- #
def _preprocess_npz(
    npz_path: Path,
    ev_key: str,
    depth_key: str,
    depth_mask_key: Optional[str],
    center_crop_hw: Tuple[int, int],
    normalize_events: bool,
    min_depth: float,
    max_depth: float,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    返回:
        ev_t: (C, H, W) float32 - 已归一化、已 center_crop 的事件张量
        gt_depth_m: (H, W) float32 - 真值深度，米；inf/超界处填 max_depth
        valid_mask: (H, W) bool - 真实有效像素 mask
    """
    data = np.load(str(npz_path), allow_pickle=True)

    ev = data[ev_key]
    ev_t = torch.from_numpy(np.asarray(ev)).float()
    if ev_t.dim() == 2:
        ev_t = ev_t.unsqueeze(0)
    ch, cw = center_crop_hw
    ev_t = center_crop_tensor_2d(ev_t, ch, cw)
    if normalize_events:
        ev_t = normalize_events_nonzero_channels(ev_t)

    depth_np = np.asarray(data[depth_key], dtype=np.float32)
    finite = np.isfinite(depth_np)
    depth_safe = np.where(finite, np.clip(depth_np, min_depth, max_depth), max_depth)
    depth_t = torch.from_numpy(depth_safe).float()
    if depth_t.dim() == 2:
        depth_t = depth_t.unsqueeze(0)
    depth_t = center_crop_tensor_2d(depth_t, ch, cw)

    if depth_mask_key and depth_mask_key in data.files:
        mask = np.asarray(data[depth_mask_key], dtype=bool)
        mask_t = torch.from_numpy(mask).bool()
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
    else:
        mask_t = (
            torch.from_numpy(finite).bool()
            & (torch.from_numpy(depth_safe) > min_depth)
            & (torch.from_numpy(depth_safe) < max_depth)
        )
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
    mask_t = center_crop_tensor_2d(mask_t, ch, cw)

    return ev_t, depth_t[0].cpu().numpy(), mask_t[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/zs/Research3/evdepth/EventDepth/629p50pn/checkpoints/epoch=022-step=38295-val_rmse=6.70.ckpt",
    )
    parser.add_argument("--config", type=str, default="config_liosam.yaml")
    parser.add_argument("--dataset_root", type=str, default="/home/zs/Research3/dataset")
    parser.add_argument("--scenes", type=str, nargs="+", default=["03", "13"])
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--no_video", action="store_true", help="若设置，则只保存 npy 和 PNG，不合成视频"
    )
    args = parser.parse_args()

    # ---------------- 加载配置与模型 ----------------
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 推理覆写：训练用 patch_crop_hw=[112,112] 配套的 in_res_hw=[112,112]，
    # 但推理输入仍为 center_crop_hw=[256,344]。网络全卷积，权重直接复用，
    # 只需把 input_padder 的目标尺寸改大到能容纳 256x344。
    ds_cfg = config["dataset"]
    center_crop_hw = tuple(int(v) for v in ds_cfg["center_crop_hw"])
    config["model"]["backbone"]["in_res_hw"] = list(center_crop_hw)
    print(f"[INFO] 推理覆写 in_res_hw -> {center_crop_hw}")
    normalize_events = bool(ds_cfg.get("normalize_events_nonzero", False))
    ev_key = ds_cfg.get("ev_key", DEFAULT_EV_KEY)
    depth_key = ds_cfg.get("depth_key", DEFAULT_DEPTH_KEY)
    depth_mask_key = ds_cfg.get("depth_mask_key", DEFAULT_DEPTH_MASK_KEY)
    depth_range = ds_cfg.get("depth_range", {})
    min_depth = float(depth_range.get("min", 0.5))
    max_depth = float(depth_range.get("max", 80.0))

    interval_sec = float(ds_cfg.get("frame_interval_sec", 0.005))
    interval_dev = float(ds_cfg.get("max_interval_deviation_sec", 0.003))
    min_dt = interval_sec - interval_dev
    max_dt = interval_sec + interval_dev

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[INFO] 加载 checkpoint: {args.ckpt}")
    model = DepthModule(config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model = model.to(device).eval()

    log_min = math.log(min_depth)
    log_max = math.log(max_depth)
    log_denom = max(log_max - log_min, 1e-6)

    # ---------------- 逐场景推理 ----------------
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        scene_path = Path(args.dataset_root) / scene
        if not scene_path.is_dir():
            print(f"[WARN] 场景目录不存在: {scene_path}，跳过")
            continue

        entries = load_liosam_index(scene_path)
        n_frames = len(entries)
        print(f"\n[SCENE {scene}] {n_frames} 帧，开始推理")

        out_pred_dir = output_root / scene / "preds"
        out_ev_dir = output_root / scene / "viz_ev"
        out_pm_dir = output_root / scene / "viz_pred_mask"
        out_pf_dir = output_root / scene / "viz_pred_full"
        out_gt_dir = output_root / scene / "viz_gt"
        for d in (out_pred_dir, out_ev_dir, out_pm_dir, out_pf_dir, out_gt_dir):
            d.mkdir(parents=True, exist_ok=True)

        prev_states = None  # 跨场景时清空
        prev_ts = None
        ch, cw = center_crop_hw
        composed_h = ch
        composed_w = cw * 4
        ffmpeg_proc = None
        video_path = output_root / scene / "video.mp4"
        if not args.no_video:
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                print("[WARN] 未找到 ffmpeg，将跳过视频合成")
            else:
                ffmpeg_proc = subprocess.Popen(
                    [
                        ffmpeg_bin, "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{composed_w}x{composed_h}",
                        "-r", str(args.fps),
                        "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        str(video_path),
                    ],
                    stdin=subprocess.PIPE,
                )

        with torch.no_grad():
            for i, (frame_id, ts, fname) in enumerate(entries):
                # 时间不连续 → 重置隐状态
                if prev_ts is not None:
                    dt = ts - prev_ts
                    if not (min_dt <= dt <= max_dt):
                        prev_states = None
                prev_ts = ts

                ev_t, gt_depth_m, valid_mask = _preprocess_npz(
                    npz_path=scene_path / fname,
                    ev_key=ev_key,
                    depth_key=depth_key,
                    depth_mask_key=depth_mask_key,
                    center_crop_hw=center_crop_hw,
                    normalize_events=normalize_events,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                ev_in = ev_t.unsqueeze(0).to(device=device, dtype=torch.float32)
                ev_in = model.input_padder.pad_tensor_ev_repr(ev_in)
                predictions, _, prev_states = model(
                    event_tensor=ev_in,
                    previous_states=prev_states,
                    retrieve_depth=True,
                    targets=None,
                    masks=None,
                )
                pred = _finest_depth_pred(predictions)  # (1,1,H',W') in norm_log
                if pred.shape[-2:] != (ch, cw):
                    pred = torch.nn.functional.interpolate(
                        pred, size=(ch, cw), mode="bilinear", align_corners=False
                    )
                pred = torch.clamp(pred, 0.0, 1.0)
                pred_log_depth = pred * log_denom + log_min
                pred_depth_m = torch.exp(pred_log_depth)[0, 0].cpu().numpy().astype(np.float32)

                stem = f"{frame_id:06d}"
                np.save(out_pred_dir / f"{stem}.npy", pred_depth_m)

                ev_img = _ev_repr_to_img_white(ev_t.cpu().numpy())
                pred_mask_img = _depth_to_colormap(
                    pred_depth_m, vmin=min_depth, vmax=max_depth, mask=valid_mask
                )
                pred_full_img = _depth_to_colormap(
                    pred_depth_m, vmin=min_depth, vmax=max_depth, mask=None
                )
                gt_img = _depth_to_colormap(
                    gt_depth_m, vmin=min_depth, vmax=max_depth, mask=valid_mask
                )

                Image.fromarray(ev_img).save(out_ev_dir / f"{stem}.png")
                Image.fromarray(pred_mask_img).save(out_pm_dir / f"{stem}.png")
                Image.fromarray(pred_full_img).save(out_pf_dir / f"{stem}.png")
                Image.fromarray(gt_img).save(out_gt_dir / f"{stem}.png")

                if ffmpeg_proc is not None:
                    composed = np.concatenate(
                        [ev_img, pred_mask_img, pred_full_img, gt_img], axis=1
                    )
                    try:
                        ffmpeg_proc.stdin.write(composed.tobytes())
                    except BrokenPipeError:
                        print("[WARN] ffmpeg 进程提前退出，视频可能不完整")
                        ffmpeg_proc = None

                if (i + 1) % 200 == 0 or (i + 1) == n_frames:
                    print(f"  [{scene}] {i + 1}/{n_frames} 帧已处理")

        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ret = ffmpeg_proc.wait()
            if ret == 0:
                print(f"[SCENE {scene}] 视频已保存: {video_path}")
            else:
                print(f"[SCENE {scene}] ffmpeg 退出码 {ret}，视频可能不完整: {video_path}")

    print("\n[DONE] 全部场景推理完成")


if __name__ == "__main__":
    main()
