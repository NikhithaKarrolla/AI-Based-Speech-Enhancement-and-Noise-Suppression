import torch

from src.models.unet import SpeechEnhancementUNet


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():

    print("=" * 60)
    print("U-NET MODEL INSPECTION")
    print("=" * 60)

    model = SpeechEnhancementUNet()

    # Parameter counts
    total_parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

    print("\nTotal parameters:")
    print(f"{total_parameters:,}")

    print("\nTrainable parameters:")
    print(f"{trainable_parameters:,}")

    # Approximate model size
    model_size_mb = (
        total_parameters * 4
    ) / (1024 ** 2)

    print("\nApproximate FP32 model size:")
    print(f"{model_size_mb:.2f} MB")

    # Input test
    x = torch.randn(
        1,
        1,
        257,
        256
    )

    with torch.no_grad():
        output = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nModel architecture:")
    print(model)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()