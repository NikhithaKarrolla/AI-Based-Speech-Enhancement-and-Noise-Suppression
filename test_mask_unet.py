import torch

from src.models.mask_unet import MaskUNet


def main():

    model = MaskUNet()

    x = torch.randn(
        2,
        1,
        257,
        256
    )

    output = model(x)

    print("Input shape:")
    print(x.shape)

    print("Output shape:")
    print(output.shape)

    print(
        "Minimum mask value:",
        output.min().item()
    )

    print(
        "Maximum mask value:",
        output.max().item()
    )

    assert output.shape == x.shape

    assert output.min().item() >= 0.0

    assert output.max().item() <= 1.0

    print()
    print("SUCCESS!")


if __name__ == "__main__":
    main()