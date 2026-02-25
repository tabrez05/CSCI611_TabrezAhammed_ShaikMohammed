# Report outline (5–6 pages PDF)

Use this structure for your **report.pdf**. Place the PDF in this folder and submit it on Canvas with your GitHub repo link.

---

1. **Model architecture and training setup**
   - Diagram or bullet description of CNN layers (Conv 3→32, 32→64, MaxPool, Conv 64→128, MaxPool, Flatten, FC→256, FC→10).
   - Loss: CrossEntropyLoss. Optimizer: Adam. Learning rate: 1e-3. Batch size: 64. Epochs: 20.
   - Regularization: dropout 0.25, weight decay 1e-4. Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip (train only).

2. **Training results**
   - Final test accuracy (from notebook).
   - Figure: Training loss vs epoch and Validation loss vs epoch (use `outputs/plots/loss_curves.png`). Label and reference in text.

3. **Feature map visualization (early layer)**
   - Figures: 3 test images + ≥8 first-conv feature maps each (use `outputs/plots/feature_maps_*.png`). Label and reference.
   - Discussion: what patterns do early filters detect (edges, color blobs, textures)? How do different channels respond to the same input?

4. **Maximally activating images**
   - State: “We use **max** over spatial locations as the activation score.” (or mean, if you change the notebook.)
   - Figures: Layer conv2, 3 filters, top 5 images per filter (use `outputs/plots/max_activating_*.png`). Label and reference.
   - Discussion: common visual patterns among top images; general vs class-specific filters.

5. **Brief discussion and reflection**
   - What you learned, limitations, what you’d try next.

**Checklist:** All figures labeled and referenced; 5–6 pages; PDF in this folder; submit PDF on Canvas + GitHub link in comments.
