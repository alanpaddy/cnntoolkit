FROM pytorch/pytorch:latest
RUN pip install torchvision matplotlib onnx onnxruntime
WORKDIR /app
COPY . /app
CMD ["python", "training/train_template.py"]
