# Lane/Drivable-Area Segmentation Trainer

Small DeepLabV3 training script for BDD100K-style lane/drivable-area masks. Masks are mapped to two classes (background vs target ids).

## Setup
1. Install Python 3.10+.
2. Install deps (CPU/GPU builds as appropriate):
   ```bash
   pip install -r requirements.txt
   ```
3. Place images under `10k/train` and masks under `labels/train` (or override paths via config/CLI).

## Running
```bash
python main.py
```

Override key options via CLI, e.g.:
```bash
python main.py --image-dir 10k/train --mask-dir labels/train --target-class-ids 1,2 --use-pretrained-backbone --checkpoint-path artifacts/run1.pth
```

Key config knobs (editable in code or via CLI):
- `image_dir` / `mask_dir`: directories with matching file stems.
- `target_class_ids`: mask label ids treated as positive (default `(2,)` for drivable area).
- `output_size`: resized `(H, W)` resolution.
- `use_pretrained_backbone`: set to `True` to pull torchvision weights (requires network/cache).
- `checkpoint_path`: saved model weights (directories are auto-created).

## Data expectations
- Mask file names must share the same stem as images (e.g., `img123.jpg` + `img123.png`).
- Masks are read as single-channel integer labels; resizing uses nearest-neighbor to preserve ids.
- Missing masks are skipped with a warning; at least 2 paired samples are required.

## Notes
- Training/early stopping monitor validation IoU; checkpoints write to `artifacts/` by default.
- Seeding is applied for repeatable splits (seed `42` by default).
