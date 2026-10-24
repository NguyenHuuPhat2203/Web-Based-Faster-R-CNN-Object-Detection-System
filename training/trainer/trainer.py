"""Minimal Faster R-CNN training loop with TensorBoard logging."""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.config import TrainConfig
from ml.models.faster_rcnn import create_faster_rcnn, get_device, load_state_dict
from training.trainer.seed import set_seed

logger = structlog.get_logger()


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """Return the mean per-image detection loss of ``model`` over ``dataloader``.

    torchvision detection models only return losses in train mode; that is safe
    here because the backbone uses frozen batch norm and the heads have no dropout.
    The previous train/eval mode is restored on exit.

    Raises:
        ValueError: If ``dataloader`` yields no samples.
    """
    was_training = model.training
    model.train()
    total = 0.0
    count = 0
    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = sum(model(images, targets).values())
        total += float(loss) * len(images)
        count += len(images)
    model.train(was_training)
    if count == 0:
        raise ValueError("evaluation dataloader is empty")
    return total / count


def train_faster_rcnn(
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: TrainConfig,
    *,
    output_dir: Path,
    resume_from_checkpoint: Path | None = None,
) -> Path:
    """Train Faster R-CNN and save the best checkpoint by validation loss.

    Args:
        dataloader: Training dataloader yielding ``(images, targets)``.
        val_dataloader: Validation dataloader; its mean loss picks the checkpoint.
        config: Shared train configuration.
        output_dir: Directory for checkpoints and TensorBoard logs.
        resume_from_checkpoint: Optional ``.pth`` / ``.safetensors`` whose weights
            initialize the model (optimizer state is not restored).

    Returns:
        Path to the best ``.pth`` checkpoint written under ``output_dir``.

    Raises:
        FileNotFoundError: If ``resume_from_checkpoint`` is given but missing.
    """
    set_seed(config.seed, deterministic=False)
    device = get_device()
    model = create_faster_rcnn(num_classes=config.model.num_classes).to(device)

    if resume_from_checkpoint is not None:
        if not resume_from_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint: {resume_from_checkpoint}")
        load_state_dict(model, resume_from_checkpoint, device=device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    accumulation_steps = config.gradient_accumulation_steps
    num_batches = len(dataloader)
    best_val_loss = float("inf")
    best_path = output_dir / "best_model.pth"
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())

            scaler.scale(loss / accumulation_steps).backward()
            # Also step on the last batch so a partial accumulation window is kept.
            if (step + 1) % accumulation_steps == 0 or step + 1 == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        train_loss = running / max(num_batches, 1)
        val_loss = evaluate_loss(model, val_dataloader, device)
        logger.info(
            "epoch_complete", epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss
        )
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("val/epoch_loss", val_loss, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("checkpoint_saved", path=str(best_path), val_loss=val_loss)

    writer.close()
    return best_path
