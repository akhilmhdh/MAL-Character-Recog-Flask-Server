from network import Net
from PIL import Image
import torch
from torch import nn
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import io
from PIL import Image
from torchvision import transforms, models
import base64

app = Flask(__name__)
CORS(app)

model_dict = torch.load('model/malayalamOCRv3Resnet.pt',
                        map_location=lambda storage, loc: storage)

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(512, 48), nn.LogSoftmax(dim=1))

model.load_state_dict(model_dict["model"])


def transform_image(image_bytes):
    transformations = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes)
    return transformations(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    model.eval()
    outputs = model(tensor)
    _, pred = outputs.max(1)
    return pred.item()


@app.route('/')
def hello():
    return 'Hello World!'


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        canvas = request.json["canvas"].split(',')[1]
        with open("canvas.png", "wb") as f:
            f.write(base64.decodebytes(canvas.encode()))
            f.close()
        character = get_prediction("./canvas.png")
        print(character)
        return jsonify({"alphabet": character})
