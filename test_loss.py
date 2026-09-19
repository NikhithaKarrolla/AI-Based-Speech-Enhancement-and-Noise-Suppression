import torch

from src.training.loss import SpectrogramMSELoss


loss_function = SpectrogramMSELoss()


predicted = torch.randn(
    16,
    1,
    257,
    256
)

target = torch.randn(
    16,
    1,
    257,
    256
)


loss = loss_function(predicted, target)


print("=" * 60)
print("LOSS FUNCTION TEST")
print("=" * 60)

print("\nPredicted shape:")
print(predicted.shape)

print("\nTarget shape:")
print(target.shape)

print("\nLoss:")
print(loss.item())


if loss.ndim == 0 and torch.isfinite(loss):

    print("\nSUCCESS!")
    print("MSE loss is working correctly.")

else:

    print("\nERROR!")
    print("Invalid loss value.")