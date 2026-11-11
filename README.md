# Braille OCR: reading Braille from a phone photo

DMET 901 Computer Vision, Team 41 (German University in Cairo).

A phone photo of a Braille page goes in; every Braille cell is detected and classified into one of
63 dot patterns by a **YOLO11-s detector re-implemented in TensorFlow/Keras**, and the cells are joined
into text with grade-1 Braille tables.

| | Val (49 book pages) | Test (50 real user uploads) |
|---|---|---|
| Char F1, Ovodov's protocol (IoU >= 0.5, right class) | **0.9993** | 0.852 |
| Char F1, centre-in-box matching (label-style independent) | 0.9994 | **0.955** (with scale/lighting-aware inference) |
| mAP@0.5 | 0.9938 | 0.791 |

Published baseline (Ovodov 2021, RetinaNet): char F1 0.998. See [docs/WORK_SUMMARY.md](docs/WORK_SUMMARY.md)
for the full story, including why the two test numbers differ.

## Repository layout

```
├── app/                  Local web app: upload a photo, get boxes + text
│   ├── server.py
│   └── index.html
├── docs/
│   ├── WORK_SUMMARY.md   What was done, results and findings (plain English)
│   └── milestone1/       Milestone 1 slides (pptx + pdf)
├── scripts/              Entry points (run from the repository root)
│   ├── download_data.sh       clone Angelina + build data/index.json
│   ├── prepare_data.py        CSV labels -> data/index.json (+ dataset statistics)
│   ├── train.py               train YOLO11-s
│   ├── evaluate.py            val/test metrics, per-class F1, confusion, examples
│   ├── plots.py               training curves, per-class F1, top confusions
│   ├── visualize_reading.py   HTML report: model text vs ground-truth text
│   └── preprocess_experiments.py  contrast/brightness/CLAHE study on unlabeled photos
├── src/                  Library code
│   ├── yolo11.py         YOLO11-s architecture (C3k2, SPPF, C2PSA, PAN, DFL head)
│   ├── loss.py           Task-aligned assigner + BCE + CIoU + DFL
│   ├── data.py           page loading, label-safe augmentation (no flips), 640 crops
│   ├── metrics.py        inference, NMS, mAP, char P/R/F1, centre matching
│   ├── inference.py      scale- and lighting-aware page reading (Phase-1 improvement)
│   └── braille_text.py   cells -> lines -> text (grade-1 RU/EN tables, number & capital signs)
├── results/
│   ├── baseline/         config.json, log.csv, train_log.txt, best.weights.h5, eval_plain/, eval_auto/
│   ├── figures/          augmentation check
│   ├── preprocess/       preprocessing study (results.csv, variants_sheet.jpg)
│   └── reading_check/    reading_check.html (5 pages: boxes, decoded text, diff)
├── data/                 (git-ignored) dataset, created by scripts/download_data.sh
├── HANDOFF.md            Technical hand-off notes for whoever continues the work
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/download_data.sh                      # ~125 MB, Angelina Braille dataset v1.1
python3 app/server.py                              # then open http://127.0.0.1:8000
```

Train and evaluate (RTX 2060 6 GB: ~2 min XLA compile, then ~16 s/epoch):

```bash
python3 scripts/train.py --name baseline --epochs 300
python3 scripts/evaluate.py --name baseline          # plain inference
python3 scripts/evaluate.py --name baseline --auto   # scale + lighting-aware inference
python3 scripts/visualize_reading.py --name baseline # results/reading_check/reading_check.html
```

## Data and credits

- **Angelina Braille Images Dataset v1.1**: I. G. Ovodov, "Optical Braille Recognition Using Object
  Detection Neural Network", ICCV Workshops 2021. <https://github.com/IlyaOvodov/AngelinaDataset>.
  The Braille letter tables in `src/braille_text.py` are taken from its `src/letters.py`.
- **English Braille book photos** (planned, not yet used): V. A. Zuev, SPbPU 2021,
  <https://www.kaggle.com/datasets/valeryzuev/unlabeled-braille-photos>.
- YOLO11 architecture: Ultralytics (2024); re-implemented here from the published configuration.
