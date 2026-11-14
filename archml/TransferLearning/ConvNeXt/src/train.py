import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tempfile import TemporaryDirectory


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.savefig(results_dir / "training_loss.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, [accuracy * 100 for accuracy in train_accuracies], label="Train Accuracy")
    ax.plot(epochs, [accuracy * 100 for accuracy in val_accuracies], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    fig.savefig(results_dir / "training_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def train_model(model, criterion, optimizer, scheduler, dataloaders,
                dataset_sizes, device, num_epochs=25):
    since = time.time()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    model.features.eval()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.item())
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.item())

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    return model
