import copy

import torch
import torch.nn as nn
from torch.ao.quantization import (
    convert,
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare,
    prepare_qat,
)
from torchvision.models.quantization import resnet18 as quantizable_resnet18


def build_quantizable_model(num_classes, device='cpu'):
    model = quantizable_resnet18(weights='IMAGENET1K_V1', quantize=False)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)


def load_trained_weights(trained_model, num_classes, device='cpu'):
    quant_model = quantizable_resnet18(weights=None, quantize=False)
    num_ftrs = quant_model.fc.in_features
    quant_model.fc = nn.Linear(num_ftrs, num_classes)
    quant_model.load_state_dict(trained_model.state_dict())
    return quant_model.to(device)


def to_fp16(model):
    return copy.deepcopy(model).half()


def static_quantize(model, calibration_loader, backend='fbgemm'):
    model = copy.deepcopy(model).to('cpu')
    model.eval()
    model.fuse_model()

    model.qconfig = get_default_qconfig(backend)
    prepare(model, inplace=True)

    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model(inputs.to('cpu'))

    convert(model, inplace=True)
    return model


def prepare_qat_model(model, backend='fbgemm'):
    model = copy.deepcopy(model).to('cpu')
    model.train()
    model.fuse_model()

    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    return model


def convert_qat_model(qat_model):
    qat_model = copy.deepcopy(qat_model).to('cpu').eval()
    convert(qat_model, inplace=True)
    return qat_model
