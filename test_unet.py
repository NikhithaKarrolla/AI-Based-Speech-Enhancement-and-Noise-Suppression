import torch

from src.models.unet import SpeechEnhancementUNet


def main():

    print("=" * 60)
    print("U-NET MODEL TEST")
    print("=" * 60)

    # Create model
    model = SpeechEnhancementUNet()

    print("\nModel created successfully.")

    # Dummy spectrogram
    x = torch.randn(
        2,
        1,
        257,
        256
    )

    print("\nInput shape:")
    print(x.shape)

    # Forward pass
    with torch.no_grad():

        output = model(x)

    print("\nOutput shape:")
    print(output.shape)

    # Verify dimensions
    if output.shape == x.shape:

        print("\nSUCCESS!")
        print("Input and output shapes match.")

    else:

        print("\nERROR!")
        print("Input and output shapes do not match.")


if __name__ == "__main__":
    main()