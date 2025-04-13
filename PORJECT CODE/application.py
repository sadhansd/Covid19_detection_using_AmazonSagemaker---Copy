from flask import Flask,render_template,request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2


application = Flask(__name__, static_folder='static')

MODEL_PATH = 'Covid-19_X-ray_diagnosis.h5'
model = load_model(MODEL_PATH)

img_size = 100

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

@application.route('/')
def index_view():
    return render_template('index.html')

UPLOAD_FOLDER = os.path.join('static', 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@application.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = read_image(file_path)
        prediction = model.predict(img)
        pred = np.argmax(prediction)
        if (pred == 0):
            output = "NEGATIVE"
        else:
            output = "POSITIVE"
        return render_template('index.html', result = output, user_image=file_path)
    else:
        return "Unable to read the file. Please check file extension"

if __name__  == '__main__':
    application.run(debug=True)