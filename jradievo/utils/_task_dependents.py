from types import SimpleNamespace

from PIL import Image
import polars as pl
import torch

from jradievo.datasets._simple import SimpleDataset


def get_data(
    cfg: SimpleNamespace,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Image.Image]]:
    print(f"Using data: {cfg.data}")
    data = {
        "select500": _get_translated_select500(cfg),
    }[cfg.data]
    data = _add_img_path(cfg, data)
    if cfg.no_findings is not None:
        data = _filter_no_findings(cfg, data)
    data = _add_target(cfg, data)
    if cfg.num_data is not None:
        data = data.head(cfg.num_data)
    if not cfg.load_memory:
        return data
    imgs = {row["dicom_id"]: Image.open(row["img_path"]) for row in data.rows(named=True)}
    return data, imgs


def _get_translated_select500(cfg: SimpleNamespace) -> pl.DataFrame:
    data = pl.read_csv(cfg.path.data_dir / "metadata_select500.csv")
    data = data.with_columns(pl.col("path").alias("img_path"))
    return data


def _filter_no_findings(cfg: SimpleNamespace, data: pl.DataFrame) -> pl.DataFrame:
    with open(cfg.path.data_dir / f"{cfg.no_findings}.txt") as f:
        no_findings = f.read().splitlines()
    data = data.with_columns(
        (
            "p" + pl.col("subject_id").cast(pl.String) + "/s" + pl.col("study_id").cast(pl.String)
        ).alias("subject_study"),
    )
    data = data.filter(~pl.col("subject_study").is_in(no_findings))
    print("filtered data:", len(data))
    return data


def _add_img_path(cfg: SimpleNamespace, data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        (cfg.path.img_dir.as_posix() + "/" + pl.col("dicom_id") + ".jpg").alias("img_path"),
    )
    return data


def _add_target(cfg: SimpleNamespace, data: pl.DataFrame) -> pl.DataFrame:
    target = {
        "ja_findings": "04_extracted_JAPANESE_findings",
        "ja_impression": "04_extracted_JAPANESE_impression",
        "ja_all": "translated_files",
        "ja_select500": "translated_files_select500",
    }[cfg.target]
    data = data.with_columns(
        (
            cfg.path.data_dir.as_posix()
            + "/"
            + target
            + "/p"
            + pl.col("subject_id").cast(pl.String).str.slice(0, 2)
            + "/p"
            + pl.col("subject_id").cast(pl.String)
            + "/s"
            + pl.col("study_id").cast(pl.String)
            + ".txt"
        ).alias("target_path"),
    )

    def read_txt(row):
        with open(row[0]) as f:
            d = f.read()
        if cfg.extract_target == "findings":
            d = d.replace("所見:", "所見：").replace("印象:", "印象：")
            try:
                d = d.split("所見：")[1].split("印象：")[0].strip()
            except:
                print("Extracting target failed")
                d = ""
        return d

    target = data.select([pl.col("target_path")]).map_rows(read_txt).rename({"map": "target"})

    data = pl.concat([data, target], how="horizontal")

    return data


def get_dataset(
    cfg: SimpleNamespace,
    data: pl.DataFrame | tuple[pl.DataFrame, dict[str, Image.Image]],
) -> torch.utils.data.Dataset:
    print(f"Using dataset: {cfg.dataset}")
    if cfg.dataset == "simple":
        return SimpleDataset(cfg, data)
    else:
        raise ValueError(f"Invalid dataset: {cfg.dataset}")
