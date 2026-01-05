from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch

config = PatchTSTConfig(
    num_input_channels=11,
    context_length=60,
    prediction_length=1,
    patch_length=16,
    patch_stride=8,
)
model = PatchTSTForPrediction(config)
print('Config:')
print(f'  num_input_channels: {config.num_input_channels}')
print(f'  context_length: {config.context_length}')

# Try batch, sequence, channel shape
x1 = torch.randn(2, 60, 11)  # batch, seq, channels
try:
    out = model(past_values=x1)
    print(f'Shape (batch, seq, channel) works: {x1.shape}')
except Exception as e:
    print(f'Shape (batch, seq, channel) FAILS: {str(e)[:100]}')

# Try batch, channel, sequence shape
x2 = torch.randn(2, 11, 60)  # batch, channels, seq
try:
    out = model(past_values=x2)
    print(f'Shape (batch, channel, seq) works: {x2.shape}')
except Exception as e:
    print(f'Shape (batch, channel, seq) FAILS: {str(e)[:100]}')

