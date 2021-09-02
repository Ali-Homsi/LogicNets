from models import AtrialFibrillationConv1dNeqModel
import torch

# "jsc-s": {
#     "hidden_layers": [64, 32, 32, 32],
#     "input_bitwidth": 2,
#     "hidden_bitwidth": 2,
#     "output_bitwidth": 2,
#     "input_fanin": 3,
#     "hidden_fanin": 3,
#     "output_fanin": 3,
#     "weight_decay": 1e-3,
#     "batch_size": 1024,
#     "epochs": 1000,
#     "learning_rate": 1e-3,
#     "seed": 2,
#     "checkpoint": None,
# },

model_cfg = {}
model_cfg['input_length'] = 1
model_cfg['output_length'] = 2
model_cfg['input_bitwidth'] = 2
model_cfg['hidden_bitwidth'] = 2
model_cfg['output_bitwidth'] = 2
model_cfg['input_fanin'] = 3
model_cfg['hidden_fanin'] = 3
model_cfg['output_fanin'] = 3


model = AtrialFibrillationConv1dNeqModel(model_cfg)

# print(data.shape) # torch.Size([1024, 1, 5250])
x = torch.randn(5250)
print(x)
print(x.shape) # torch.Size([5250])
input()

x = x.view(-1,1,5250)
print(x)
print(x.shape) # torch.Size([1, 1, 5250])
input()

out = model(x)
print(out)
print(out.shape) # torch.Size([1, 32, 41])
input()

