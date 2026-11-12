from pathlib import Path

import torch

from src.download_data import download_politicians
from src.dataset import split_train_val, predprocessing_data
from src.model_builder import initialize_model
from src.train import train_model


DOWNLOAD_DATA = True
SPLIT_DATA = True

BATCH_SIZE = 4
NUM_EPOCHS = 25


def main():
    if DOWNLOAD_DATA:
        download_politicians()

    if SPLIT_DATA:
        split_train_val(val_ratio=0.2, seed=42)

    data_transforms, dataloaders, dataset_sizes, class_names, device = (
        predprocessing_data(batch_size=BATCH_SIZE)
    )

    print("Классы:", class_names)
    print("Размер выборок:", dataset_sizes)

    model_conv, criterion, optimizer_conv, exp_lr_scheduler = (
        initialize_model(
            device,
            num_classes=len(class_names)
        )
    )

    model_conv = train_model(
        model_conv,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        device,
        num_epochs=NUM_EPOCHS
    )

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model_conv.state_dict(),
            "class_names": class_names,
        },
        checkpoint_dir / "convnext_base.pth"
    )

    print("Модель успешно сохранена!")


if __name__ == "__main__":
    main()
