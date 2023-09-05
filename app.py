from flask import Flask, request, jsonify, render_template
import torch
import json
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sklearn
import joblib
import os

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/instruction')
def instruction_page():
    return render_template('instruction.html')

@app.route('/image-upload')
def image_upload():
    return render_template('image-upload.html')

class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=16 * 16 * 32, out_features=128)  # Update in_features
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.lr = joblib.load(filename='logistic.pkl')

    def forward(self, x, genetics=0, age='empty', sex='empty', location='empty'):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 32)  # Update the size here
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x)
        if age != 'empty' and sex != 'empty' and location != 'empty':
            lr = self.lr.predict_proba([[age, sex, location]])
            if genetics == 1:
                lr[0][0] -= 0.10
                lr[0][1] += 0.10
            if lr[0][1] > lr[0][0]:
                x[0][0] -= 0.05
                x[0][1] += 0.05
        else:
            if genetics == 1:
                x[0][0] -= 0.05
                x[0][1] += 0.05
        return x


# Load the trained model
model = SkinCancerCNN(num_classes=2)
model.load_state_dict(torch.load('skin_cancer_model.pth'))
model.eval()


# Preprocess incoming image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image


@app.route('/predict', methods=['POST'])
def predict():
    classify = {0: 'It is predicted to be a benign growth, please bear in mind that this is not a formal diagnosis!', 1: 'It is possible to be a cancerous growth, please bear in mind that this is not a formal diagnosis!'}
    try:
        genetics = int(request.args.get('genetics', 0))
        age = request.args.get('age', 'empty')
        sex = request.args.get('sex', 'empty')
        location = request.args.get('location', 'empty')
        if 'file' not in request.files:
            return render_template('error.html')

        file = request.files['file']
        if file.filename == '':
            return render_template('error.html')

        if file:
            if genetics == 'Yes':
                genetics = 1
            if sex != 'empty':
                with open('sex.json', 'r') as sex_file:
                    sex_dict = json.load(sex_file)
                    sex = sex_dict[sex]
            if location != 'empty':
                with open('loc.json', 'r') as loc_file:
                    loc_dict = json.load(loc_file)
                    location = loc_dict[location]
            image = Image.open(file)
            image_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(image_tensor, genetics, age, sex, location)
                probabilities = F.softmax(output[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                result_text = classify.get(predicted_class, 'unknown')
            return render_template('result.html', result=result_text)
    except Exception as e:
        return render_template('error.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
