import torch
from models.resnet_custom import get_resnet18

model = get_resnet18()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx", input_names=['input'], output_names=['output'])
