import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


@dataclass
class TrainingConfig:
    image_dir: str = "10k/train"
    mask_dir: str = "labels/train"
    output_size: Tuple[int, int] = (256, 512)  # (H, W)
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    num_classes: int = 2
    patience: int = 5
    sample_cap: Optional[int] = 200
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0
    target_class_ids: Tuple[int, ...] = (2,)
    checkpoint_path: str = "artifacts/deeplabv3_bdd100k.pth"
    use_pretrained_backbone: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_paths(cfg: TrainingConfig) -> None:
    if not Path(cfg.image_dir).exists():
        raise FileNotFoundError(f"Image directory not found: {cfg.image_dir}")
    if not Path(cfg.mask_dir).exists():
        raise FileNotFoundError(f"Mask directory not found: {cfg.mask_dir}")
    checkpoint_dir = Path(cfg.checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_image_transforms(size: Tuple[int, int]) -> T.Compose:
    return T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def encode_mask(mask_array: np.ndarray, target_class_ids: Sequence[int]) -> torch.Tensor:
    mask = np.isin(mask_array, target_class_ids).astype(np.int64)
    # Keep mapping explicit to avoid silent label drift if upstream masks change. @Rtur2003
    return torch.from_numpy(mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on lane/drivable masks.")
    parser.add_argument("--image-dir", type=str, help="Directory with training images.")
    parser.add_argument("--mask-dir", type=str, help="Directory with label masks.")
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Resize target (height width).",
    )
    parser.add_argument("--batch-size", type=int, help="Mini-batch size.")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
    parser.add_argument("--patience", type=int, help="Early stopping patience.")
    parser.add_argument("--sample-cap", type=int, help="Limit dataset size for quick runs.")
    parser.add_argument(
        "--val-split",
        type=float,
        help="Validation split fraction between 0 and 1.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count.")
    parser.add_argument(
        "--target-class-ids",
        type=str,
        help="Comma-separated mask ids to keep as foreground (e.g., '1,2').",
    )
    parser.add_argument(
        "--use-pretrained-backbone",
        action="store_true",
        help="Load torchvision pretrained backbone (requires cached weights or network).",
    )
    parser.add_argument("--checkpoint-path", type=str, help="Where to save weights.")
    parser.add_argument(
        "--device",
        type=str,
        help="Device override: 'cuda' or 'cpu'.",
    )
    return parser.parse_args()


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.image_dir:
        cfg.image_dir = args.image_dir
    if args.mask_dir:
        cfg.mask_dir = args.mask_dir
    if args.output_size:
        cfg.output_size = (args.output_size[0], args.output_size[1])
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.num_epochs:
        cfg.num_epochs = args.num_epochs
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.patience:
        cfg.patience = args.patience
    if args.sample_cap is not None:
        cfg.sample_cap = args.sample_cap
    if args.val_split:
        cfg.val_split = args.val_split
    if args.seed:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.target_class_ids:
        cfg.target_class_ids = tuple(int(x) for x in args.target_class_ids.split(",") if x.strip())
    if args.use_pretrained_backbone:
        cfg.use_pretrained_backbone = True
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.device:
        cfg.device = args.device
    return cfg
class BDD100KDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        target_class_ids: Sequence[int],
        output_size: Tuple[int, int],
        sample_cap: Optional[int],
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.target_class_ids = tuple(target_class_ids)
        self.output_size = output_size
        self.transforms = get_image_transforms(output_size)
        self.pairs = self._collect_pairs(sample_cap)

    @staticmethod
    def _list_files(directory: Path) -> List[Path]:
        return sorted(p for p in directory.iterdir() if p.is_file())

    def _collect_pairs(self, sample_cap: Optional[int]) -> List[Tuple[Path, Path]]:
        image_files = self._list_files(self.image_dir)
        mask_files = self._list_files(self.mask_dir)
        mask_map = {p.stem: p for p in mask_files}

        pairs: List[Tuple[Path, Path]] = []
        missing_masks: List[str] = []
        for img in image_files:
            mask = mask_map.get(img.stem)
            if mask:
                pairs.append((img, mask))
            else:
                missing_masks.append(img.name)

        if not pairs:
            raise ValueError("No matching image/mask pairs found in provided directories.")
        if missing_masks:
            print(f"Warning: {len(missing_masks)} images skipped because masks are missing.")

        if sample_cap is not None:
            pairs = pairs[:sample_cap]
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transforms(image)
        mask = mask.resize((self.output_size[1], self.output_size[0]), Image.NEAREST)
        mask_array = np.array(mask, dtype=np.int64)
        binary_mask = encode_mask(mask_array, self.target_class_ids)

        return image, binary_mask


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.bool()
    targets = targets.bool()
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def get_deeplab_model(cfg: TrainingConfig) -> torch.nn.Module:
    weights = None
    if cfg.use_pretrained_backbone:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    try:
        return deeplabv3_resnet50(weights=weights, num_classes=cfg.num_classes)
    except Exception as exc:
        if weights is not None:
            print(f"Falling back to random init for segmentation head: {exc}")
            return deeplabv3_resnet50(weights=None, num_classes=cfg.num_classes)
        raise


def create_dataloaders(dataset: Dataset, cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    if len(dataset) < 2:
        raise ValueError("Dataset needs at least 2 samples to create a validation split.")

    val_size = max(1, int(len(dataset) * cfg.val_split))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)

    batches = max(1, len(loader))
    return total_loss / batches, total_iou / batches


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)

    batches = max(1, len(loader))
    return total_loss / batches, total_iou / batches


def train(cfg: TrainingConfig) -> None:
    set_seed(cfg.seed)
    ensure_paths(cfg)

    dataset = BDD100KDataset(
        cfg.image_dir,
        cfg.mask_dir,
        cfg.target_class_ids,
        cfg.output_size,
        cfg.sample_cap,
    )

    train_loader, val_loader = create_dataloaders(dataset, cfg)
    device = torch.device(cfg.device)
    model = get_deeplab_model(cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_iou = 0.0
    epochs_no_improve = 0

    print(
        f"Starting training on {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val samples"
    )

    for epoch in range(cfg.num_epochs):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train loss {train_loss:.4f} IoU {train_iou:.4f} | "
            f"val loss {val_loss:.4f} IoU {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f"Saved checkpoint to {cfg.checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No val IoU improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= cfg.patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    args = parse_args()
    config = apply_overrides(TrainingConfig(), args)
    train(config)
