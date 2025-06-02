import os.path as osp
from typing import List, Optional

import mmcv
import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmseg.registry import DATASETS
from mmseg.utils import get_classes, get_palette
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class RailSem19Dataset(BaseSegDataset):
    """RailSem19Dataset dataset.

    In segmentation map annotation for RailSem19Dataset,
    ``reduce_zero_label`` is fixed to False. The ``img_suffix``
    is fixed to '.jpg' and ``seg_map_suffix`` is fixed to '.png'.

    Args:
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
    """

    METAINFO = dict(
        classes=(
            "background",
            "rail",
            "rail_switch",
            "rail_crossing",
            "rail_switch_stand",
            "rail_switch_stand_light",
            "rail_switch_stand_light_off",
            "rail_switch_stand_light_on",
            "rail_switch_stand_light_off_on",
            "rail_switch_stand_light_on_off",
            "rail_switch_stand_light_off_on_off",
            "rail_switch_stand_light_on_off_on",
            "rail_switch_stand_light_off_on_off_on",
            "rail_switch_stand_light_on_off_on_off",
            "rail_switch_stand_light_off_on_off_on_off",
            "rail_switch_stand_light_on_off_on_off_on",
            "rail_switch_stand_light_off_on_off_on_off_on",
            "rail_switch_stand_light_on_off_on_off_on_off",
        )
    )

    def __init__(self, img_suffix=".jpg", seg_map_suffix=".png", **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get("img_path")
        ann_dir = self.data_prefix.get("seg_map_path")
        file_backend = get_file_backend(img_dir)

        for img_name in file_backend.list_dir_or_file(
            dir_path=img_dir, list_dir=False, suffix=self.img_suffix, recursive=True
        ):
            data_info = dict(
                img_path=osp.join(img_dir, img_name),
                seg_map_path=osp.join(
                    ann_dir, img_name.replace(self.img_suffix, self.seg_map_suffix)
                ),
            )
            data_info["reduce_zero_label"] = self.reduce_zero_label
            data_info["seg_fields"] = []
            data_list.append(data_info)

        data_list = sorted(data_list, key=lambda x: x["img_path"])
        return data_list

    def get_classes(self, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return self.METAINFO["classes"]
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmengine.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def get_palette(self, palette=None):
        """Get palette of current dataset.

        Args:
            palette (Sequence[Sequence[int]]] | np.ndarray | None): If
                palette is None, use default PALETTE defined by builtin dataset.
                If palette is a sequence of colors, override the PALETTE
                defined by the dataset.

        Returns:
            list[tuple[int]] or np.ndarray: The palette of this dataset.
        """
        if palette is None:
            return self.METAINFO["palette"]
        elif isinstance(palette, list):
            return palette
        elif isinstance(palette, tuple):
            return list(palette)
        elif isinstance(palette, np.ndarray):
            return palette.tolist()
        else:
            raise ValueError(f"Unsupported type {type(palette)} of palette.")
