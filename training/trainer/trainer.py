"""Minimal Faster R-CNN training loop with TensorBoard logging."""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.config import TrainConfig
from ml.models.faster_rcnn import create_faster_rcnn, get_device
from training.trainer.seed import set_seed

logger = structlog.get_logger()


def train_faster_rcnn(
    dataloader: DataLoader,
    config: TrainConfig,
    *,
    output_dir: Path,
    resume_from_checkpoint: Path | None = None,
) -> Path:
    """Train Faster R-CNN and save the best checkpoint by training loss.

    Args:
        dataloader: Training dataloader yielding ``(images, targets)``.
        config: Shared train configuration.
        output_dir: Directory for checkpoints and TensorBoard logs.
        resume_from_checkpoint: Optional ``.pth`` / ``.safetensors`` to resume.

    Returns:
        Path to the best ``.pth`` checkpoint written under ``output_dir``.
    """
    set_seed(config.seed, deterministic=False)
    device = get_device()
    model = create_faster_rcnn(num_classes=config.model.num_classes).to(device)

    if resume_from_checkpoint and resume_from_checkpoint.exists():
        from ml.models.faster_rcnn import load_state_dict

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

    best_loss = float("inf")
    best_path = output_dir / "best_model.pth"
    global_step = 0

    model.train()
    for epoch in range(config.epochs):
        running = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * config.gradient_accumulation_steps
            steps += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        avg_loss = running / max(steps, 1)
        logger.info("epoch_complete", epoch=epoch + 1, loss=avg_loss)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            logger.info("checkpoint_saved", path=str(best_path), loss=best_loss)

    writer.close()
    return best_path
