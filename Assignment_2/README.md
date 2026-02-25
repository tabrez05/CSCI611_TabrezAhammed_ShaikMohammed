# CSCI 611 Assignment 2 — CNN on CIFAR-10

Task 1: Train CNN, loss curves, test accuracy. Task 2A: First-conv feature maps. Task 2B: Maximally activating images.

## Requirements

- Python 3.8+
- PyTorch, torchvision, matplotlib, numpy

```bash
pip install -r requirements.txt
```

## How to run (choose one)

### Option 1: Python script (recommended — one command)

From the `Assignment_2` folder:

```bash
python run_cnn.py
```

Optional arguments:

```bash
python run_cnn.py --data_dir ./data --out_dir outputs --epochs 20 --batch_size 64 --seed 42
```

### Option 2: Jupyter notebook

1. Open `build_cnn.ipynb`.
2. Select a kernel that has PyTorch (e.g. `.venv` if you use the project venv).
3. Run the first cell; it runs `run_cnn.py` and produces all outputs.

## Configuration

Edit the `CONFIG` dict at the top of **`run_cnn.py`** to change:

- `data_dir`, `out_dir`, `batch_size`, `epochs`, `val_frac`, `lr`, `weight_decay`
- Task 2A: `num_feature_maps` (default 8)
- Task 2B: `task2b_layer_index`, `task2b_filters`, `task2b_activation` ("mean" or "max"), `task2b_top_k`

Or pass overrides from the command line: `--data_dir`, `--out_dir`, `--epochs`, `--batch_size`, `--seed`.

## Outputs

All written to `outputs/` (or `--out_dir`):

- `best_model.pt` — best model state and training history
- `loss_curves.png` — train/val loss vs epoch
- `task2A_featuremaps_img1.png`, `img2.png`, `img3.png` — first-conv feature maps for 3 images
- `task2B_top5_filter3.png`, `filter11.png`, `filter29.png` — top-5 activating images per filter

## Report (LaTeX)

A full report template is in \texttt{report.tex}. To compile:

1. Run \texttt{python run\_cnn.py} so \texttt{outputs/} contains all figures.
2. From \texttt{Assignment_2/}: \texttt{pdflatex report.tex} (run twice if needed for references).
3. Replace \texttt{[Your Name]} and insert your final test accuracy in the report.
4. Submit the resulting \texttt{report.pdf} on Canvas.

Figure paths in the .tex point to \texttt{outputs/loss\_curves.png}, \texttt{outputs/task2A\_featuremaps\_img*.png}, and \texttt{outputs/task2B\_top5\_filter*.png}.

## Repo structure (for submission)

```
Assignment_2/
├── run_cnn.py         # Main script (run this)
├── build_cnn.ipynb    # Notebook that runs run_cnn.py
├── report.tex         # LaTeX source for the report
├── README.md
├── requirements.txt
├── outputs/           # Created when you run (figures + best_model.pt)
└── report.pdf         # Compiled report (submit on Canvas)
```

Submit on Canvas: PDF report + GitHub repo link. Make repo public or invite instructor (GitHub: boshen-csuchico).
