from pathlib import Path

import os
import torch
import shutil
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def split_train_val(val_ratio=0.2, seed=42):
    data_dir = (
        Path(__file__).resolve().parent.parent
        / "data/politicians"
    )

    raw_dir = data_dir / "raw"

    splits = {}

    for person_dir in sorted(raw_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        images = sorted(person_dir.glob("*.jpg"))

        if len(images) < 2:
            raise ValueError(
                f"Недостаточно изображений: {person_dir.name}"
            )

        train_images, val_images = train_test_split(
            images,
            test_size=val_ratio,
            random_state=seed
        )

        splits[person_dir.name] = (train_images, val_images)

    if not splits:
        raise ValueError("Датасет пуст")

    for split in ["train", "val"]:
        output_dir = data_dir / split

        if output_dir.exists():
            shutil.rmtree(output_dir)

    for person, (train_images, val_images) in splits.items():
        for split, images in [
            ("train", train_images),
            ("val", val_images)
        ]:
            output_dir = data_dir / split / person
            output_dir.mkdir(parents=True, exist_ok=True)

            for image in images:
                shutil.copy2(image, output_dir / image.name)

        print(
            f"{person}: "
            f"{len(train_images)} train / "
            f"{len(val_images)} val"
        )


def predprocessing_data(batch_size=4):
    data_dir = (
            Path(__file__).resolve().parent.parent
            / "data/politicians"
            )

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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cpu"
    print(f"Using {device} device")


    return data_transforms, dataloaders, dataset_sizes, class_names, device
