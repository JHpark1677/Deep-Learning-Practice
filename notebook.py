import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50 # , ResNet50_Weights

# backbone = nn.ModuleList(resnet50(pretrained=True).children())

# print(backbone)

# print("*************************")

# for name, parameter in backbone.named_parameters():
#     print('name : ', name)

# query_embed = nn.Embedding(10, 1)
# print(query_embed.weight)

# print("*******************")

# print(query_embed.weight.repeat(3, 1, 1))
torch.cuda.nccl.is_available(torch.randn(1).cuda())
torch.cuda.nccl.version()
