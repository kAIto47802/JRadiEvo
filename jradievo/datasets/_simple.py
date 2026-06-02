from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import polars as pl
import torch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: SimpleNamespace,
        data: pl.DataFrame | tuple[pl.DataFrame, dict[str, Image.Image]],
    ) -> None:
        self._cfg = cfg
        if isinstance(data, tuple):
            self._data, self._imgs = data
        else:
            self._data, self._imgs = data, None

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Image.Image, str]:
        if self._imgs is not None:
            img = self._imgs[self._data[int(idx), "dicom_id"]]
        else:
            img_path = self._data[int(idx), "img_path"]
            img = Image.open(img_path)
        img = img.resize(self._cfg.img_size)
        dicom_id = self._data[int(idx), "dicom_id"]
        return self._data[int(idx), "target"], img, dicom_id
