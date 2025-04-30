# 🧠 CNNToolkit – Everything You Need for Convolutional Neural Networks

Modular, production-ready utilities including prototyping, training, deployment and explainability.

## 🧩 Features
- Ready-to-use CNNs: ResNet, U-Net, EfficientNet
- Training loops, eval metrics, dataset loaders
- Deployment tools: ONNX export, quantization
- Grad-CAM and saliency map explainability
- Lightweight Docker setup for repeatability

## 🚀 Quickstart

```bash
docker build -t cnntoolkit .
docker run -it cnntoolkit python training/train_template.py
