import torch

from src.models.mask_unet import MaskUNet


def main():

    model = MaskUNet()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    model_size_mb = (
        total_parameters * 4
    ) / (
        1024 ** 2
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Approximate FP32 model size:",
        f"{model_size_mb:.2f} MB"
    )

    x = torch.randn(
        1,
        1,
        257,
        256
    )

    with torch.no_grad():

        output = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )


if __name__ == "__main__":
    main()