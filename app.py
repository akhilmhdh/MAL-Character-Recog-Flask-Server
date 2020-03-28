from network import Net
from PIL import Image
import torch
from flask import Flask, jsonify, request
import requests
import io
import torchvision.transforms as transform

app = Flask(__name__)

model_dict = torch.load('model/malayalamOCR.pt',
                        map_location=lambda storage, loc: storage)
model = Net()
model.load_state_dict(model_dict["model"])


def transform_image(image_bytes):
    transformations = transform.Compose([transform.Grayscale(1),
                                         transform.Resize(32),
                                         transform.ToTensor(),
                                         transform.Normalize([0.5], [0.5])
                                         ])
    image = Image.open(io.BytesIO(image_bytes))
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
        file = request.files['file']
        img_bytes = file.read()
        alphabet = get_prediction(image_bytes=img_bytes)
        return jsonify({'network': "CNN", 'class_label': alphabet})


# resp = requests.post("https://cr-mal.herokuapp.com/predict",
#                      files={"file": open('alphabet_test1.png', 'rb')})
# print(resp)
