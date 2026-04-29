import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler


def initialize_model(num_classes, device):
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model_conv.fc.parameters(),
                               lr=0.001,
                               momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv,
                                           step_size=7,
                                           gamma=0.1)
    return model_conv, criterion, optimizer_conv, exp_lr_scheduler
