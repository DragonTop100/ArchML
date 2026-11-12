import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "checkpoints/convnext_base.pth"
VAL_DIR = BASE_DIR / "data/politicians/val"
RESULT_DIR = BASE_DIR / "results"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = convnext_base(weights=None)

    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        num_features, len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(model, image, class_names, device):
    transform = ConvNeXt_Base_Weights.DEFAULT.transforms()

    inputs = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return (
        class_names[prediction.item()],
        confidence.item()
    )


def visualize_predictions(paths, model, class_names, device):
    fig, axes = plt.subplots(
        1, len(paths),
        figsize=(5 * len(paths), 4),
        squeeze=False
    )

    for ax, path in zip(axes[0], paths):
        with Image.open(path) as source:
            image = source.convert("RGB")

        prediction, confidence = predict_image(
            model, image, class_names, device
        )

        ax.imshow(image)
        ax.set_title(
            f"{prediction}\nConfidence: {confidence:.1%}"
        )
        ax.axis("off")

        print(f"{path.name}: {prediction} ({confidence:.1%})")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULT_DIR / "predictions.png"
    fig.savefig(output_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"Результат сохранён: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--random", type=int)

    args = parser.parse_args()

    if args.image is not None:
        paths = [args.image.expanduser().resolve()]
        if not paths[0].is_file():
            parser.error("Изображение не найдено")
    else:
        if args.random < 1:
            parser.error("Количество должно быть положительным")

        images = sorted(
            p for p in VAL_DIR.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if len(images) < args.random:
            parser.error("Недостаточно изображений в val")

        paths = random.sample(images, args.random)

    model, class_names, device = load_model()

    visualize_predictions(
        paths, model, class_names, device
    )


if __name__ == "__main__":
    main()
