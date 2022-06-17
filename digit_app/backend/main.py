from numpy.core.fromnumeric import ptp
from digit_app import app
from flask import make_response, jsonify, request

import base64
import io
from PIL import Image
import numpy  as np
import json
import os 
from posixpath import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class MLP(nn.Module):
    
    def __init__(self, input_size=28*28, num_classes=10):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes        
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes) 
        self.dropout = nn.Dropout(0.0)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)    
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )
    
    def forward(self, x):
        return self.model(x)



def _inference(image, model, preprocess):
    with torch.no_grad():
        pred = model(torch.unsqueeze(preprocess(image), axis=0).float())
        return F.softmax(pred, dim=-1).cpu()


def predict_digit(image, _model):

    root_path = os.path.dirname(__file__)

    if _model == "mlp":
        model = MLP()
        model_path = os.path.join(root_path, "models", "mlp_model.pt")
    elif _model == "lenet":
        model = Lenet5()
        model_path = os.path.join(root_path, "models", "lenet_model.pt")
    else:
        print("Unknown model")


    # preprocessing 
    preprocess = transforms.Compose([transforms.ToTensor()])

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))    
    model.eval()

    prediction = _inference(image, model, preprocess)
    print("model_output:", prediction)
    
    pred_digit = torch.argmax(prediction)
    pred_prob = torch.max(prediction) * 100
    print("predicted:", pred_digit.item())
    print("confidence: {:.2f} %".format(pred_prob.item()))
    
    return pred_digit, pred_prob.type(torch.int16)


def makeResponse():
    json_data = {}
    req = request.get_json()
    img = req["image"][22:]
    model = req["model"]
    
    img = base64.b64decode(img)
    img = io.BytesIO(img)
    img =  Image.open(img)

    # convert image RGBA -> RGB -> Gray
    img = img.convert("RGB") 
    img = img.convert("L")

    # resize image
    img_resized = np.asarray(img.resize((28,28), resample=Image.NEAREST))

    img_resized = 1 - img_resized/255
    prediction, probability = predict_digit(img_resized, model)

    # response back to the client from server
    json_data["server_message"] = "It is all good!"
    json_data["prediction"] = json.dumps(prediction.numpy().tolist())
    json_data["probability"] = json.dumps(probability.numpy().tolist())
    
    return json_data



@app.route("/recognize_digit/get_info", methods=["POST"])
def digitRecognizeServerResponse():
    return make_response(jsonify(makeResponse()), 200)
