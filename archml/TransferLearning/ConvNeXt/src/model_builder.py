import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision.models import resnet18, ResNet18_Weights


def initialize_model(device, num_classes):
    model_conv = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
    # model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.classifier[2].in_features
    model_conv.classifier[2] = nn.Linear(
        num_ftrs, num_classes
    )
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, num_classes)
    # object.__setattr__(model_conv, "features", nn.Sequential(*list(model_conv.children())[:-2]))

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(
        model_conv.classifier[2].parameters(),
        lr=0.001,
        momentum=0.9
    )
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv,
        step_size=10,
        gamma=0.5
    )

    return model_conv, criterion, optimizer_conv, exp_lr_scheduler
