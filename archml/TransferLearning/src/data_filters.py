import random
import shutil
from pathlib import Path

import cv2
import imagehash
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError


def validate_images(directory, min_width=200, min_height=200,
                    allowed_formats=None, verbose=True):
    allowed_formats = allowed_formats or {'JPEG', 'PNG'}
    directory = Path(directory)
    removed, total = 0, 0

    files = [x for x in directory.iterdir() if x.is_file()]

    for f in files:
        total += 1
        try:
            with Image.open(f) as img:
                img.verify()
            with Image.open(f) as img:
                fmt_ok = img.format in allowed_formats
                width, height = img.size
        except (UnidentifiedImageError, OSError):
            f.unlink(missing_ok=True)
            removed += 1
            continue

        if not fmt_ok or width < min_width or height < min_height:
            f.unlink(missing_ok=True)
            removed += 1

    if verbose:
        print(f'  [validate] {directory.name}: удалено {removed}/{total}')
    return removed


def remove_blurry(directory, threshold=100.0, verbose=True):
    directory = Path(directory)
    removed, total = 0, 0

    files = [x for x in directory.iterdir() if x.is_file()]

    for f in files:
        total += 1
        img = cv2.imread(str(f))
        if img is None:
            f.unlink(missing_ok=True)
            removed += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < threshold:
            f.unlink(missing_ok=True)
            removed += 1

    if verbose:
        print(f'  [blur] {directory.name}: удалено {removed}/{total}')
    return removed


def remove_duplicates(directory, hash_size=8, max_distance=5, verbose=True):
    directory = Path(directory)
    files = sorted(f for f in directory.iterdir() if f.is_file())
    kept_hashes = []
    removed = 0

    for f in files:
        with Image.open(f) as img:
            h = imagehash.phash(img, hash_size=hash_size)

        is_duplicate = any((h - kept) <= max_distance for kept in kept_hashes)
        if is_duplicate:
            f.unlink(missing_ok=True)
            removed += 1
        else:
            kept_hashes.append(h)

    if verbose:
        print(f'  [dedup] {directory.name}: удалено {removed}, '
              f'осталось {len(kept_hashes)}')
    return removed


def _build_mtcnn(device):
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device=device)


def filter_by_faces(directory, min_faces=1, max_faces=1, device='cpu',
                    detector=None, verbose=True):
    directory = Path(directory)
    detector = detector or _build_mtcnn(device)
    removed, total = 0, 0

    files = [x for x in directory.iterdir() if x.is_file()]

    for f in files:
        total += 1
        img = Image.open(f).convert('RGB')

        boxes, _ = detector.detect(img)
        n_faces = 0 if boxes is None else len(boxes)
        if n_faces < min_faces or n_faces > max_faces:
            f.unlink(missing_ok=True)
            removed += 1

    if verbose:
        print(f'  [faces] {directory.name}: удалено {removed}/{total}')
    return removed


def _build_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(
            'openai/clip-vit-base-patch32'
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    return model, processor


def filter_by_clip(directory, prompt, threshold=0.25, device='cpu',
                   clip=None, verbose=True):
    directory = Path(directory)
    model, processor = clip or _build_clip(device)

    text_inputs = processor(
            text=[prompt], return_tensors='pt', padding=True
    ).to(device)
    with torch.no_grad():
        text_features = F.normalize(
                model.get_text_features(**text_inputs), dim=-1
        )

    removed, total = 0, 0
    files = [x for x in directory.iterdir() if x.is_file()]

    for f in files:
        total += 1
        img = Image.open(f).convert('RGB')

        image_inputs = processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            image_features = F.normalize(
                    model.get_image_features(**image_inputs), dim=-1
            )
        similarity = (image_features @ text_features.T).item()

        if similarity < threshold:
            f.unlink(missing_ok=True)
            removed += 1

    if verbose:
        print(f'  [clip] {directory.name}: удалено {removed}/{total}')
    return removed


def clean_dataset(raw_dir, min_width=200, min_height=200, allowed_formats=None,
                  blur_threshold=100.0, dedup_hash_size=8,
                  dedup_max_distance=5, min_faces=1, max_faces=1,
                  clip_prompt_template=None, clip_threshold=0.25,
                  device='cpu'):
    raw_dir = Path(raw_dir)

    face_detector = _build_mtcnn(device)
    clip_bundle = _build_clip(device) if clip_prompt_template else None

    person_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]

    for person_dir in person_dirs:
        display_name = person_dir.name.replace('_', ' ')
        print(f'{display_name}')

        validate_images(person_dir, min_width, min_height, allowed_formats)
        remove_blurry(person_dir, blur_threshold)
        remove_duplicates(person_dir, dedup_hash_size, dedup_max_distance)
        filter_by_faces(person_dir, min_faces, max_faces, device,
                        detector=face_detector)

        if clip_prompt_template:
            prompt = clip_prompt_template.format(name=display_name)
            filter_by_clip(person_dir, prompt, clip_threshold, device,
                           clip=clip_bundle)

        remaining = sum(1 for f in person_dir.iterdir() if f.is_file())
        print(f'  итого осталось: {remaining}\n')


def split_train_val(clean_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)

    person_dirs = [d for d in sorted(clean_dir.iterdir()) if d.is_dir()]

    for person_dir in person_dirs:
        images = [f for f in person_dir.iterdir() if f.is_file()]
        random.shuffle(images)
        val_count = int(len(images) * val_ratio)
        val_files = set(images[:val_count])

        train_out = output_dir / 'train' / person_dir.name
        val_out = output_dir / 'val' / person_dir.name
        train_out.mkdir(parents=True, exist_ok=True)
        val_out.mkdir(parents=True, exist_ok=True)

        for f in images:
            dest_dir = val_out if f in val_files else train_out
            shutil.copy2(f, dest_dir / f.name)

        print(f'{person_dir.name}: {len(images) - val_count} train / '
              f'{val_count} val')
