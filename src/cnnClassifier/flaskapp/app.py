from flask import Flask,render_template,request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2
from cnnClassifier.components.prediction import Prediction
import sys

app = Flask(__name__, static_folder='static')

current_path = os.path.dirname(os.path.abspath(__file__))

def read_image(filename):
    img = image.load_img(filename, target_size=(100, 100))
    img = np.array(img)
    if (img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray / 255
    resized = cv2.resize(gray, (img_size, img_size))
    reshaped = resized.reshape(1, img_size, img_size)
    return reshaped

@app.route('/')
def index_view():
    return render_template('index.html')

UPLOAD_FOLDER = os.path.join(f'{current_path}\static', 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = obj.process_image(file_path)
        output = obj.predict(img)
        relative_path = os.path.join('static/uploads', filename)
        return render_template('index.html', result = output, user_image=relative_path)
    else:
        return "Unable to read the file. Please check file extension"

if __name__  == '__main__':
    obj = Prediction()
    app.run(host='0.0.0.0', port=8080,debug=True)