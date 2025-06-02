import os.path as osp
import json
from typing import List

import numpy as np
from mmengine.fileio import get_file_backend
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class RailSem19Dataset(BaseSegDataset):
    """RailSem19 dataset with 19 classes and color palette from rs19-config.json."""

    # Path to the dataset root
    data_root = "data/RailSem19/"
    cfg_path = osp.join(data_root, "rs19-config.json")

    # Load class names and palette from JSON
    with open(cfg_path, "r") as f:
        json_data = json.load(f)
        METAINFO = {
            "classes": [item["name"] for item in json_data["labels"]],
            "palette": [tuple(item["color"]) for item in json_data["labels"]],
        }

    def __init__(self, img_suffix=".jpg", seg_map_suffix=".png", **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
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

