from pathlib import Path

import os
import random
import torch
import shutil
from functools import partial
from PIL import Image
from stable_datasets.images import FMD
from torchvision import transforms
from sklearn.model_selection import train_test_split


IMAGES_PER_CLASS = 100

def load_fmd():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    dataset = FMD(
        split="train",
        download_dir=data_dir,
        processed_cache_dir=data_dir / "processed",
    )
    class_names = dataset.features["label"].names
    records = list(dataset.remove_columns(["image", "mask"]))
    rng = random.Random(42)
    sample_rng = random.Random(42)
    indices = {"train": [], "val": []}
    val_count = round(IMAGES_PER_CLASS * 0.2)
    train_count = IMAGES_PER_CLASS - val_count

    for label, name in enumerate(class_names):
        class_indices = {"train": [], "val": []}
        groups = ["final"] if name == "foliage" else ["moderate", "object"]
        for group in groups:
            images = sorted(
                (i for i, row in enumerate(records)
                 if row["label"] == label and f"/{name}_{group}_" in row["image_id"]),
                key=lambda i: records[i]["image_id"],
            )
            expected = 100 if group == "final" else 50
            if len(images) != expected:
                raise ValueError(f"Неожиданное число файлов {name}/{group}: {len(images)}")
            rng.shuffle(images)
            group_val_count = len(images) // 5
            class_indices["val"].extend(images[:group_val_count])
            class_indices["train"].extend(images[group_val_count:])

        if IMAGES_PER_CLASS < 100:
            class_indices["train"] = sample_rng.sample(class_indices["train"], train_count)
            class_indices["val"] = sample_rng.sample(class_indices["val"], val_count)
        indices["train"].extend(class_indices["train"])
        indices["val"].extend(class_indices["val"])

    dataset = dataset.remove_columns("mask")
    datasets = {split: dataset.select(indices[split]) for split in indices}

    for split in ["train", "val"]:
        split_dir = data_dir / split
        expected_files = {
            (split_dir / records[index]["image_id"]).with_suffix(".png")
            for index in indices[split]
        }
        if split_dir.exists():
            extra_files = {path for path in split_dir.rglob("*") if path.is_file()} - expected_files
            if extra_files:
                raise ValueError(
                    f"В {split_dir} есть лишние файлы; проверьте содержимое перед повторной подготовкой"
                )

    for split, subset in datasets.items():
        for sample in subset:
            output = (data_dir / split / sample["image_id"]).with_suffix(".png")
            image = sample["image"].convert("RGB")
            if output.exists():
                with Image.open(output) as existing:
                    if existing.convert("RGB").tobytes() != image.tobytes() or existing.size != image.size:
                        raise ValueError(f"Существующий файл отличается от FMD: {output}")
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                image.save(output)

    print(f"FMD: train — {len(indices['train'])}, validation — {len(indices['val'])} "
          f"({train_count}/{val_count} на класс)")
    return datasets, class_names


def transform_sample(sample, transform):
    return transform(sample["image"].convert("RGB")), sample["label"]


def predprocessing_data(batch_size=4):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    datasets, class_names = load_fmd()
    image_datasets = {
        x: datasets[x].remove_columns("image_id").with_transform(
            partial(transform_sample, transform=data_transforms[x])
        ) for x in ['train', 'val']
    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cpu"
    print(f"Using {device} device")


    return data_transforms, dataloaders, dataset_sizes, class_names, device
