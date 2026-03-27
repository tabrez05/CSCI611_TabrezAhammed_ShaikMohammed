# CSCI 611 — Assignment 3 (YOLO small-object traffic-sign detection)

**Student:** Tabrez Ahammed Shaik Mohammed  
**ID:** 012623820  
**Institution:** California State University, Chico  

Repository path: **`Assignment_3/`** in [CSCI611_TabrezAhammed_ShaikMohammed](https://github.com/tabrez05/CSCI611_TabrezAhammed_ShaikMohammed).

## Contents (GitHub submission)

| Item | Description |
|------|-------------|
| `assignment3_yolo.ipynb` | Main Colab-ready notebook: Roboflow download, baseline, fine-tuning, resolution ablation, NMS sweep, exports. |
| `Assignment3_Report.pdf` | Compiled written report (submit this PDF unless your instructor asks for LaTeX sources only). |
| `report/Assignment3_Report.tex` | LaTeX source for the report. |
| `report/figures/` | Figures referenced by the report (`fig-*.png`; optional `cell*.png` notebook exports). |
| `report/COMPILE.txt` | How to rebuild the PDF from `.tex`. |

## Not included (by design)

- **Dataset images/labels** — Download at runtime via Roboflow API (see notebook `RUN_CFG` and API key setup).
- **Pretrained weights** — `yolov8n.pt` is fetched automatically by Ultralytics.
- **Training runs / checkpoints** — Regenerated when you execute the notebook on GPU.

## Quick check before submit

1. Open `assignment3_yolo.ipynb` and confirm outputs are present (or re-run on Colab).
2. Open `Assignment3_Report.pdf` and verify figures render.
3. Do **not** commit secrets: keep Roboflow API keys in Colab secrets / env vars only.
