from network import Net
from PIL import Image
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import io
from PIL import Image
import torchvision.transforms as transform
import base64

app = Flask(__name__)
CORS(app)

model_dict = torch.load('model/malayalamOCR.pt',
                        map_location=lambda storage, loc: storage)
model = Net()
model.load_state_dict(model_dict["model"])


def transform_image(image_bytes):
    transformations = transform.Compose([
        transform.Grayscale(1),
        transform.Resize(32),
        transform.ToTensor(),
        transform.Normalize([0.5], [0.5])
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
