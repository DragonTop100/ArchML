import os
import random

import torch
from torchvision import datasets, transforms


def predprocessing_data(data_dir='../data/politicians', batch_size=4,
                        subset_fraction=1.0, seed=42):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_datasets = {}
    for split in ['train', 'val']:
        full_datasets[split] = datasets.ImageFolder(
            os.path.join(data_dir, split),
            data_transforms[split],
            allow_empty=True
        )
    class_names = full_datasets['train'].classes

    image_datasets = dict(full_datasets)
    if subset_fraction < 1.0:
        random.seed(seed)
        n_total = len(full_datasets['train'])
        n_subset = max(1, int(n_total * subset_fraction))
        indices = random.sample(range(n_total), n_subset)
        image_datasets['train'] = torch.utils.data.Subset(
            full_datasets['train'], indices
        )

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cpu"
    print(f"Using {device} device")

    return data_transforms, dataloaders, dataset_sizes, class_names, device
