# JRadiEvo: A Japanese Radiology Report Generation Model Enhanced by Evolutionary Optimization of Model Merging

## Installation

```bash
git clone https://github.com/kAIto47802/JRadiEvo.git
cd JRadiEvo

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Training

```bash
python -m jradievo.train
```

## Inference

```bash
python -m jradievo.inference
```


## Evaluation

```bash
python -m jradievo.evaluate --input-path path/to/generated/reports.csv
```


## Citation
```bibtex
@article{baba2024jradievo
    title={{JRadiEvo}: A Japanese Radiology Report Generation Model Enhanced by Evolutionary Optimization of Model Merging},
    author={Kaito Baba and Ryota Yagi and Junichiro Takahashi and Risa Kishikawa and Satoshi Kodera},
    journal={arXiv preprint arXiv:2411.09933},
    year={2024}
}
