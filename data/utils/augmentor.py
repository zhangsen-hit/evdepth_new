import collections.abc as abc
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from warnings import filterwarnings, warn

import torch as th
import torch.distributions.categorical
from typing import Dict, Any
from torch.nn.functional import interpolate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.types import DataType, LoaderDataDictGenX
from utils.helpers import torch_uniform_sample_scalar

NO_LABEL_WARN_MSG = 'No Labels found. This can lead to a crash and should not happen often.'
filterwarnings('always', message=NO_LABEL_WARN_MSG)


class SpatialAugmentorPatchCrop:
    """序列级 patch 裁剪 + 水平翻转（参考 rpg_e2depth）。

    训练：每次 __call__ 随机采样 (y0, x0) 与是否 h-flip；同一次调用对所有帧
    使用相同的窗口与翻转，从而保证一个序列内时序一致。
    验证/测试：固定中心裁剪，不翻转。

    输入数据为 LoaderDataDictGenX，期望含 DataType.EV_REPR（List[Tensor (C,H,W)]），
    可选 DataType.DEPTH / DataType.DEPTH_MASK（List[Tensor (1,H,W)]）。
    """

    def __init__(self,
                 crop_hw: Tuple[int, int],
                 training: bool,
                 h_flip_prob: float = 0.5):
        ch, cw = int(crop_hw[0]), int(crop_hw[1])
        assert ch > 0 and cw > 0
        self.crop_h = ch
        self.crop_w = cw
        self.training = bool(training)
        self.h_flip_prob = float(h_flip_prob) if self.training else 0.0
        assert 0.0 <= self.h_flip_prob <= 1.0

    def _sample_window(self, in_h: int, in_w: int) -> Tuple[int, int, bool]:
        assert in_h >= self.crop_h and in_w >= self.crop_w, \
            f'input ({in_h}x{in_w}) smaller than crop ({self.crop_h}x{self.crop_w})'
        if self.training:
            y0 = int(th.randint(0, in_h - self.crop_h + 1, (1,)).item())
            x0 = int(th.randint(0, in_w - self.crop_w + 1, (1,)).item())
            do_flip = th.rand(1).item() < self.h_flip_prob
        else:
            y0 = (in_h - self.crop_h) // 2
            x0 = (in_w - self.crop_w) // 2
            do_flip = False
        return y0, x0, do_flip

    def _apply(self, t: th.Tensor, y0: int, x0: int, do_flip: bool) -> th.Tensor:
        out = t[..., y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if do_flip:
            out = th.flip(out, dims=[-1])
        return out

    def __call__(self, item: LoaderDataDictGenX) -> LoaderDataDictGenX:
        ev_list = item.get(DataType.EV_REPR, None)
        if ev_list is None or len(ev_list) == 0:
            return item
        in_h, in_w = ev_list[0].shape[-2:]
        y0, x0, do_flip = self._sample_window(int(in_h), int(in_w))

        item[DataType.EV_REPR] = [self._apply(t, y0, x0, do_flip) for t in ev_list]
        if DataType.DEPTH in item:
            item[DataType.DEPTH] = [self._apply(t, y0, x0, do_flip) for t in item[DataType.DEPTH]]
        if DataType.DEPTH_MASK in item:
            item[DataType.DEPTH_MASK] = [self._apply(t, y0, x0, do_flip) for t in item[DataType.DEPTH_MASK]]
        return item


@dataclass
class ZoomOutState:
    active: bool
    x0: int
    y0: int
    zoom_out_factor: float


@dataclass
class RotationState:
    active: bool
    angle_deg: float


@dataclass
class AugmentationState:
    apply_h_flip: bool
    rotation: RotationState
    apply_zoom_in: bool
    zoom_out: ZoomOutState


class RandomSpatialAugmentorGenX:
    def __init__(self,
                 dataset_hw: Tuple[int, int],
                 automatic_randomization: bool,
                 augm_config: Dict[str, Any]):
        assert isinstance(dataset_hw, tuple)
        assert len(dataset_hw) == 2
        assert all(x > 0 for x in dataset_hw)
        assert isinstance(automatic_randomization, bool)

        self.hw_tuple = dataset_hw
        self.automatic_randomization = automatic_randomization

        # 只启用水平翻转；旋转和缩放一律关闭，避免对深度/几何造成不必要的扰动
        if isinstance(augm_config, dict):
            self.h_flip_prob = augm_config.get('prob_hflip', 0.5)
        else:
            self.h_flip_prob = 0.5

        # 强制关闭旋转与缩放相关的概率（即使配置里给了非零值也忽略）
        self.rot_prob = 0.0
        self.rot_min_angle_deg = 0.0
        self.rot_max_angle_deg = 0.0

        self.zoom_prob = 0.0
        # 仍然初始化 zoom 相关参数，但不会被实际使用
        zoom_out_weight = 1.0
        self.min_zoom_out_factor = 1.0
        self.max_zoom_out_factor = 1.0
        zoom_in_weight = 0.0
        self.min_zoom_in_factor = 1.0
        self.max_zoom_in_factor = 1.0

        assert 0 <= self.h_flip_prob <= 1
        assert 0 <= self.rot_prob <= 1
        assert 0 <= self.rot_min_angle_deg <= self.rot_max_angle_deg
        assert 0 <= self.zoom_prob <= 1
        assert 0 <= zoom_in_weight
        assert self.max_zoom_in_factor >= self.min_zoom_in_factor >= 1
        assert 0 <= zoom_out_weight
        assert self.max_zoom_out_factor >= self.min_zoom_out_factor >= 1
        if not automatic_randomization:
            # We are probably applying augmentation to a streaming dataset for which zoom in augm is not supported.
            assert zoom_in_weight == 0, f'{zoom_in_weight=}'

        self.zoom_in_or_out_distribution = torch.distributions.categorical.Categorical(
            probs=th.tensor([zoom_in_weight, zoom_out_weight]))

        self.augm_state = AugmentationState(
            apply_h_flip=False,
            rotation=RotationState(active=False, angle_deg=0.0),
            apply_zoom_in=False,
            zoom_out=ZoomOutState(active=False, x0=0, y0=0, zoom_out_factor=1.0))

    def randomize_augmentation(self):
        """Sample new augmentation parameters that will be consistently applied among the items.

        This function only works with augmentations that are input-independent.
        E.g. The zoom-in augmentation parameters depend on the labels and cannot be sampled in this function.
        For the same reason, it is not a very reasonable augmentation for the streaming scenario.
        """
        self.augm_state.apply_h_flip = self.h_flip_prob > th.rand(1).item()

        self.augm_state.rotation.active = self.rot_prob > th.rand(1).item()
        if self.augm_state.rotation.active:
            sign = 1 if th.randn(1).item() >= 0 else -1
            self.augm_state.rotation.angle_deg = sign * torch_uniform_sample_scalar(
                min_value=self.rot_min_angle_deg, max_value=self.rot_max_angle_deg)

        # Zoom in and zoom out is mutually exclusive.
        do_zoom = self.zoom_prob > th.rand(1).item()
        do_zoom_in = self.zoom_in_or_out_distribution.sample().item() == 0
        do_zoom_out = not do_zoom_in
        do_zoom_in &= do_zoom
        do_zoom_out &= do_zoom
        self.augm_state.apply_zoom_in = do_zoom_in
        self.augm_state.zoom_out.active = do_zoom_out
        if do_zoom_out:
            rand_zoom_out_factor = torch_uniform_sample_scalar(
                min_value=self.min_zoom_out_factor, max_value=self.max_zoom_out_factor)
            height, width = self.hw_tuple
            zoom_window_h, zoom_window_w = int(height / rand_zoom_out_factor), int(width / rand_zoom_out_factor)
            x0_sampled = int(torch_uniform_sample_scalar(min_value=0, max_value=width - zoom_window_w))
            y0_sampled = int(torch_uniform_sample_scalar(min_value=0, max_value=height - zoom_window_h))
            self.augm_state.zoom_out.x0 = x0_sampled
            self.augm_state.zoom_out.y0 = y0_sampled
            self.augm_state.zoom_out.zoom_out_factor = rand_zoom_out_factor

    def _zoom_out_and_rescale(self, data_dict: LoaderDataDictGenX) -> LoaderDataDictGenX:
        zoom_out_state = self.augm_state.zoom_out

        zoom_out_factor = zoom_out_state.zoom_out_factor
        if zoom_out_factor == 1:
            return data_dict
        return {k: RandomSpatialAugmentorGenX._zoom_out_and_rescale_recursive(
            v, zoom_coordinates_x0y0=(zoom_out_state.x0, zoom_out_state.y0),
            zoom_out_factor=zoom_out_factor, datatype=k) for k, v in data_dict.items()}

    @staticmethod
    def _zoom_out_and_rescale_tensor(input_: th.Tensor,
                                     zoom_coordinates_x0y0: Tuple[int, int],
                                     zoom_out_factor: float,
                                     datatype: DataType) -> th.Tensor:
        assert len(zoom_coordinates_x0y0) == 2
        assert isinstance(input_, th.Tensor)

        if datatype == DataType.IMAGE or datatype == DataType.EV_REPR:
            assert input_.ndim == 3, f'{input_.shape=}'
            height, width = input_.shape[-2:]
            zoom_window_h, zoom_window_w = int(height / zoom_out_factor), int(width / zoom_out_factor)
            zoom_window = interpolate(input_.unsqueeze(0), size=(zoom_window_h, zoom_window_w), mode='nearest-exact')[0]
            output = th.zeros_like(input_)

            x0, y0 = zoom_coordinates_x0y0
            assert x0 >= 0
            assert y0 >= 0
            output[:, y0:y0 + zoom_window_h, x0:x0 + zoom_window_w] = zoom_window
            return output
        if datatype == DataType.DEPTH or datatype == DataType.DEPTH_MASK:
            # 深度图和掩码是 2D 张量 (H, W)
            assert input_.ndim == 2, f'Depth/mask should be 2D, got {input_.shape=}'
            height, width = input_.shape
            zoom_window_h, zoom_window_w = int(height / zoom_out_factor), int(width / zoom_out_factor)
            # 添加 batch 和 channel 维度用于 interpolate
            input_expanded = input_.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            zoom_window = interpolate(input_expanded, size=(zoom_window_h, zoom_window_w), mode='nearest-exact')
            zoom_window = zoom_window[0, 0]  # 恢复为 2D (H', W')
            output = th.zeros_like(input_)
            x0, y0 = zoom_coordinates_x0y0
            assert x0 >= 0
            assert y0 >= 0
            output[y0:y0 + zoom_window_h, x0:x0 + zoom_window_w] = zoom_window
            return output
        raise NotImplementedError

    @classmethod
    def _zoom_out_and_rescale_recursive(cls,
                                        input_: Any,
                                        zoom_coordinates_x0y0: Tuple[int, int],
                                        zoom_out_factor: float,
                                        datatype: DataType):
        if datatype in (DataType.IS_PADDED_MASK, DataType.IS_FIRST_SAMPLE):
            return input_
        if isinstance(input_, th.Tensor):
            return cls._zoom_out_and_rescale_tensor(input_=input_,
                                                    zoom_coordinates_x0y0=zoom_coordinates_x0y0,
                                                    zoom_out_factor=zoom_out_factor,
                                                    datatype=datatype)
        if isinstance(input_, ObjectLabels) or isinstance(input_, SparselyBatchedObjectLabels):
            assert datatype == DataType.OBJLABELS or datatype == DataType.OBJLABELS_SEQ
            input_.zoom_out_and_rescale_(zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_out_factor=zoom_out_factor)
            return input_
        if isinstance(input_, abc.Sequence):
            return [RandomSpatialAugmentorGenX._zoom_out_and_rescale_recursive(
                x, zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_out_factor=zoom_out_factor, datatype=datatype) \
                for x in input_]
        if isinstance(input_, abc.Mapping):
            return {key: RandomSpatialAugmentorGenX._zoom_out_and_rescale_recursive(
                value, zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_out_factor=zoom_out_factor, datatype=datatype) \
                for key, value in input_.items()}
        raise NotImplementedError

    def _zoom_in_and_rescale(self, data_dict: LoaderDataDictGenX) -> LoaderDataDictGenX:
        rand_zoom_in_factor = torch_uniform_sample_scalar(min_value=self.min_zoom_in_factor,
                                                          max_value=self.max_zoom_in_factor)
        if rand_zoom_in_factor == 1:
            return data_dict

        height, width = RandomSpatialAugmentorGenX._hw_from_data(data_dict=data_dict)
        assert (height, width) == self.hw_tuple
        zoom_window_h, zoom_window_w = int(height / rand_zoom_in_factor), int(width / rand_zoom_in_factor)
        latest_objframe = get_most_recent_objframe(data_dict=data_dict, check_if_nonempty=True)
        if latest_objframe is None:
            warn(message=NO_LABEL_WARN_MSG, category=UserWarning, stacklevel=2)
            return data_dict
        x0_sampled, y0_sampled = randomly_sample_zoom_window_from_objframe(
            objframe=latest_objframe, zoom_window_height=zoom_window_h, zoom_window_width=zoom_window_w)

        return {k: RandomSpatialAugmentorGenX._zoom_in_and_rescale_recursive(
            v, zoom_coordinates_x0y0=(x0_sampled, y0_sampled), zoom_in_factor=rand_zoom_in_factor, datatype=k) \
            for k, v in data_dict.items()}

    @staticmethod
    def _zoom_in_and_rescale_tensor(input_: th.Tensor,
                                    zoom_coordinates_x0y0: Tuple[int, int],
                                    zoom_in_factor: float,
                                    datatype: DataType) -> th.Tensor:
        assert len(zoom_coordinates_x0y0) == 2
        assert isinstance(input_, th.Tensor)

        if datatype == DataType.IMAGE or datatype == DataType.EV_REPR:
            assert input_.ndim == 3, f'{input_.shape=}'
            height, width = input_.shape[-2:]
            zoom_window_h, zoom_window_w = int(height / zoom_in_factor), int(width / zoom_in_factor)

            x0, y0 = zoom_coordinates_x0y0
            assert x0 >= 0
            assert y0 >= 0
            zoom_canvas = input_[..., y0:y0 + zoom_window_h, x0:x0 + zoom_window_w].unsqueeze(0)
            output = interpolate(zoom_canvas, size=(height, width), mode='nearest-exact')
            output = output[0]
            return output
        if datatype == DataType.DEPTH or datatype == DataType.DEPTH_MASK:
            # 深度图和掩码是 2D 张量 (H, W)
            assert input_.ndim == 2, f'Depth/mask should be 2D, got {input_.shape=}'
            height, width = input_.shape
            zoom_window_h, zoom_window_w = int(height / zoom_in_factor), int(width / zoom_in_factor)

            x0, y0 = zoom_coordinates_x0y0
            assert x0 >= 0
            assert y0 >= 0
            zoom_canvas = input_[y0:y0 + zoom_window_h, x0:x0 + zoom_window_w]
            # 添加 batch 和 channel 维度用于 interpolate
            zoom_canvas_expanded = zoom_canvas.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
            output = interpolate(zoom_canvas_expanded, size=(height, width), mode='nearest-exact')
            output = output[0, 0]  # 恢复为 2D (H, W)
            return output
        raise NotImplementedError

    @classmethod
    def _zoom_in_and_rescale_recursive(cls,
                                       input_: Any,
                                       zoom_coordinates_x0y0: Tuple[int, int],
                                       zoom_in_factor: float,
                                       datatype: DataType):
        if datatype in (DataType.IS_PADDED_MASK, DataType.IS_FIRST_SAMPLE):
            return input_
        if isinstance(input_, th.Tensor):
            return cls._zoom_in_and_rescale_tensor(input_=input_,
                                                   zoom_coordinates_x0y0=zoom_coordinates_x0y0,
                                                   zoom_in_factor=zoom_in_factor,
                                                   datatype=datatype)
        if isinstance(input_, ObjectLabels) or isinstance(input_, SparselyBatchedObjectLabels):
            assert datatype == DataType.OBJLABELS or datatype == DataType.OBJLABELS_SEQ
            input_.zoom_in_and_rescale_(zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_in_factor=zoom_in_factor)
            return input_
        if isinstance(input_, abc.Sequence):
            return [RandomSpatialAugmentorGenX._zoom_in_and_rescale_recursive(
                x, zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_in_factor=zoom_in_factor, datatype=datatype) \
                for x in input_]
        if isinstance(input_, abc.Mapping):
            return {key: RandomSpatialAugmentorGenX._zoom_in_and_rescale_recursive(
                value, zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_in_factor=zoom_in_factor, datatype=datatype) \
                for key, value in input_.items()}
        raise NotImplementedError

    def _rotate(self, data_dict: LoaderDataDictGenX) -> LoaderDataDictGenX:
        angle_deg = self.augm_state.rotation.angle_deg
        return {k: RandomSpatialAugmentorGenX._rotate_recursive(v, angle_deg=angle_deg, datatype=k)
                for k, v in data_dict.items()}

    @staticmethod
    def _rotate_tensor(input_: Any, angle_deg: float, datatype: DataType):
        assert isinstance(input_, th.Tensor)
        if datatype == DataType.IMAGE or datatype == DataType.EV_REPR:
            return rotate(input_, angle=angle_deg, interpolation=InterpolationMode.NEAREST)
        if datatype == DataType.DEPTH or datatype == DataType.DEPTH_MASK:
            # 深度图和掩码也使用 nearest 插值进行旋转
            return rotate(input_, angle=angle_deg, interpolation=InterpolationMode.NEAREST)
        raise NotImplementedError

    @classmethod
    def _rotate_recursive(cls, input_: Any, angle_deg: float, datatype: DataType):
        if datatype in (DataType.IS_PADDED_MASK, DataType.IS_FIRST_SAMPLE):
            return input_
        if isinstance(input_, th.Tensor):
            return cls._rotate_tensor(input_=input_, angle_deg=angle_deg, datatype=datatype)
        if isinstance(input_, ObjectLabels) or isinstance(input_, SparselyBatchedObjectLabels):
            assert datatype == DataType.OBJLABELS or datatype == DataType.OBJLABELS_SEQ
            input_.rotate_(angle_deg=angle_deg)
            return input_
        if isinstance(input_, abc.Sequence):
            return [RandomSpatialAugmentorGenX._rotate_recursive(x, angle_deg=angle_deg, datatype=datatype) \
                    for x in input_]
        if isinstance(input_, abc.Mapping):
            return {key: RandomSpatialAugmentorGenX._rotate_recursive(value, angle_deg=angle_deg, datatype=datatype) \
                    for key, value in input_.items()}
        raise NotImplementedError

    @staticmethod
    def _flip(data_dict: LoaderDataDictGenX, type_: str) -> LoaderDataDictGenX:
        assert type_ in {'h', 'v'}
        return {k: RandomSpatialAugmentorGenX._flip_recursive(v, flip_type=type_, datatype=k) \
                for k, v in data_dict.items()}

    @staticmethod
    def _flip_tensor(input_: Any, flip_type: str, datatype: DataType):
        assert isinstance(input_, th.Tensor)
        flip_axis = -1 if flip_type == 'h' else -2
        if datatype == DataType.IMAGE or datatype == DataType.EV_REPR:
            return th.flip(input_, dims=[flip_axis])
        if datatype == DataType.DEPTH or datatype == DataType.DEPTH_MASK:
            # 深度图和深度掩码的翻转与图像相同，只翻转空间维度
            return th.flip(input_, dims=[flip_axis])
        if datatype == DataType.FLOW:
            assert input_.shape[-3] == 2
            flow_idx = 0 if flip_type == 'h' else 1
            input_ = th.flip(input_, dims=[flip_axis])
            # Also flip the sign of the x (horizontal) or y (vertical) component of the flow.
            input_[..., flow_idx, :, :] = -1 * input_[..., flow_idx, :, :]
            return input_
        raise NotImplementedError

    @classmethod
    def _flip_recursive(cls, input_: Any, flip_type: str, datatype: DataType):
        if datatype in (DataType.IS_PADDED_MASK, DataType.IS_FIRST_SAMPLE):
            return input_
        if isinstance(input_, th.Tensor):
            return cls._flip_tensor(input_=input_, flip_type=flip_type, datatype=datatype)
        if isinstance(input_, ObjectLabels) or isinstance(input_, SparselyBatchedObjectLabels):
            assert datatype == DataType.OBJLABELS or datatype == DataType.OBJLABELS_SEQ
            if flip_type == 'h':
                # in-place modification
                input_.flip_lr_()
                return input_
            else:
                raise NotImplementedError
        if isinstance(input_, abc.Sequence):
            return [RandomSpatialAugmentorGenX._flip_recursive(x, flip_type=flip_type, datatype=datatype) \
                    for x in input_]
        if isinstance(input_, abc.Mapping):
            return {key: RandomSpatialAugmentorGenX._flip_recursive(value, flip_type=flip_type, datatype=datatype) \
                    for key, value in input_.items()}
        raise NotImplementedError

    @staticmethod
    def _hw_from_data(data_dict: LoaderDataDictGenX) -> Tuple[int, int]:
        height = None
        width = None
        for k, v in data_dict.items():
            _hw = None
            if k == DataType.OBJLABELS or k == DataType.OBJLABELS_SEQ:
                hw = v.input_size_hw
                if hw is not None:
                    _hw = v.input_size_hw
            elif k in (DataType.IMAGE, DataType.FLOW, DataType.EV_REPR):
                _hw = v[0].shape[-2:]
            if _hw is not None:
                _height, _width = _hw
                if height is None:
                    assert width is None
                    height, width = _height, _width
                else:
                    assert height == _height and width == _width
        assert height is not None
        assert width is not None
        return height, width

    def __call__(self, data_dict: LoaderDataDictGenX):
        """
        :param data_dict: LoaderDataDictGenX type, image-based tensors must have (*, h, w) shape.
        :return: map with same keys but spatially augmented values.
        """
        if self.automatic_randomization:
            self.randomize_augmentation()

        if self.augm_state.apply_h_flip:
            data_dict = self._flip(data_dict, type_='h')
        if self.augm_state.rotation.active:
            data_dict = self._rotate(data_dict)
        if self.augm_state.apply_zoom_in:
            data_dict = self._zoom_in_and_rescale(data_dict=data_dict)
        if self.augm_state.zoom_out.active:
            assert not self.augm_state.apply_zoom_in
            data_dict = self._zoom_out_and_rescale(data_dict=data_dict)
        return data_dict


def get_most_recent_objframe(data_dict: LoaderDataDictGenX, check_if_nonempty: bool = True) -> Optional[ObjectLabels]:
    assert DataType.OBJLABELS_SEQ in data_dict, f'Requires datatype {DataType.OBJLABELS_SEQ} to be present'
    sparse_obj_labels = data_dict[DataType.OBJLABELS_SEQ]
    sparse_obj_labels: SparselyBatchedObjectLabels

    for obj_label in reversed(sparse_obj_labels):
        if obj_label is not None:
            return_label = True if not check_if_nonempty else len(obj_label) > 0
            if return_label:
                return obj_label
    # no labels found
    return None


def randomly_sample_zoom_window_from_objframe(
        objframe: ObjectLabels,
        zoom_window_height: Union[int, float],
        zoom_window_width: Union[int, float]) -> Tuple[int, int]:
    input_height, input_width = objframe.input_size_hw
    possible_samples = []
    for idx in range(len(objframe)):
        label_xywh = (objframe.x[idx], objframe.y[idx], objframe.w[idx], objframe.h[idx])
        possible_samples.append(
            randomly_sample_zoom_window_from_label_rectangle(
                label_xywh=label_xywh,
                input_height=input_height,
                input_width=input_width,
                zoom_window_height=zoom_window_height,
                zoom_window_width=zoom_window_width)
        )
    assert len(possible_samples) > 0
    # Using torch to sample, to avoid potential problems with multiprocessing.
    sample_idx = 0 if len(possible_samples) == 1 else th.randint(low=0, high=len(possible_samples) - 1,
                                                                 size=(1,)).item()
    x0_sample, y0_sample = possible_samples[sample_idx]
    assert input_width > x0_sample >= 0, f'{x0_sample=}'
    assert input_height > y0_sample >= 0, f'{y0_sample=}'
    return x0_sample, y0_sample


def randomly_sample_zoom_window_from_label_rectangle(
        label_xywh: Tuple[Union[int, float, th.Tensor], ...],
        input_height: Union[int, float],
        input_width: Union[int, float],
        zoom_window_height: Union[int, float],
        zoom_window_width: Union[int, float]) -> Tuple[int, int]:
    """ Computes a set of top-left coordinates from which the top-left corner of the zoom window
    can be sampled such that the zoom window is guaranteed to contain the whole (rectangular) label.
    Return a random sample from this set.

    Notation:
    (x0,y0)---(x1,y0)
     |             |
     |             |
    (x0,y1)---(x1,y1)
    """
    assert input_height >= zoom_window_height
    assert input_width >= zoom_window_width
    label_xywh = tuple(x.item() if isinstance(x, th.Tensor) else x for x in label_xywh)
    x0_l, y0_l, w_l, h_l = label_xywh
    x1_l = x0_l + w_l
    y1_l = y0_l + h_l
    assert x0_l >= 0
    assert y0_l >= 0
    assert w_l > 0
    assert h_l > 0
    assert x1_l <= input_width + 1e-2 - 1
    assert y1_l <= input_height + 1e-2 - 1

    x0_valid_region = max(x1_l - max(zoom_window_width, w_l), 0)
    y0_valid_region = max(y1_l - max(zoom_window_height, h_l), 0)
    x1_valid_region = min(x0_l + max(zoom_window_width, w_l), input_width - 1)
    y1_valid_region = min(y0_l + max(zoom_window_height, h_l), input_height - 1)

    x1_valid_region = max(x1_valid_region - zoom_window_width, x0_valid_region)
    y1_valid_region = max(y1_valid_region - zoom_window_height, y0_valid_region)

    x_topleft_sample = int(torch_uniform_sample_scalar(min_value=x0_valid_region, max_value=x1_valid_region))
    assert 0 <= x_topleft_sample < input_width
    y_topleft_sample = int(torch_uniform_sample_scalar(min_value=y0_valid_region, max_value=y1_valid_region))
    assert 0 <= y_topleft_sample < input_height
    return x_topleft_sample, y_topleft_sample
