import io
import itertools
import time

import torch


def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def measure_inference_latency(model, dataloader, device='cpu',
                              n_batches=20, warmup=5,
                              input_dtype=torch.float32):
    model = model.to(device).eval()
    per_image_times = []
    n_measured = 0

    with torch.no_grad():
        for i, (inputs, _) in enumerate(itertools.cycle(dataloader)):
            inputs = inputs.to(device=device, dtype=input_dtype)

            if i < warmup:
                model(inputs)
                continue

            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            per_image_times.append(elapsed / inputs.size(0))
            n_measured += 1
            if n_measured >= n_batches:
                break

    if not per_image_times:
        raise ValueError('dataloader пуст — нечего измерять')

    return (sum(per_image_times) / len(per_image_times)) * 1000


def evaluate_accuracy(model, dataloader, device='cpu',
                      input_dtype=torch.float32):
    model = model.to(device).eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device=device, dtype=input_dtype)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    return correct / total


def benchmark_all(models_dict, dataloader, device='cpu', input_dtypes=None):
    input_dtypes = input_dtypes or {}
    cpu_only_methods = {'INT8_static', 'QAT'}
    results = {}

    for name, model in models_dict.items():
        dtype = input_dtypes.get(name, torch.float32)
        model_device = 'cpu' if name in cpu_only_methods else device

        size_mb = get_model_size_mb(model)
        latency_ms = measure_inference_latency(
            model, dataloader, device=model_device, input_dtype=dtype
        )
        accuracy = evaluate_accuracy(
            model, dataloader, device=model_device, input_dtype=dtype
        )

        results[name] = {
            'size_mb': size_mb,
            'latency_ms': latency_ms,
            'accuracy': accuracy,
        }
        print(f'{name}: size={size_mb:.2f} МБ, '
              f'latency={latency_ms:.2f} мс/img, acc={accuracy:.4f}')

    return results
