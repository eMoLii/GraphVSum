# GraphVSum

GraphVSum is a novel framework that represents multimodal video content as a heterogeneous graph, employing a dual fusion technique and a multimodal joint learning approach to produce coherent and semantically rich summaries.

## Requirements

You can install the conda environment by running:

```bash
pip install -r requirements.txt
```

## Datasets

- [BLiSS](https://drive.google.com/drive/folders/1rqXEIelRzq4mb7NaBk3GXxh7jlfP_Snm): This dataset comprises livestream videos from the Behance platform.
- [PlotSnap](https://github.com/katha-ai/RecapStorySumm-CVPR2024): Specically designed for long-story summarization tasks, this dataset consists of two popular crime thriller TV series: 24 and Prison Break.
- [TVSum](https://drive.google.com/drive/folders/1rqXEIelRzq4mb7NaBk3GXxh7jlfP_Snm): It comprises 50 YouTube videos spanning 10 categories.

## Running

You can run GraphVSum by executing the following command:

```bash
python -m trainer.py
