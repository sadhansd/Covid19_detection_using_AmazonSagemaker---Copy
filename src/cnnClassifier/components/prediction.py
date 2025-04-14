from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from cnnClassifier.config.configuration import ConfigurationManager
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

class Prediction:
    def __init__(self):
        self.config = ConfigurationManager()
        self.model_path = self.config.get_model_path()
        self.model = load_model(self.model_path)
        
    def process_image(self,filename):
        try:
            img_size = 100
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
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
    
    def predict(self, img):
        prediction = self.model.predict(img)
        result = np.argmax(prediction)
        if result == 0:
            return "Negative"
        else:
            return "Positive"
        
    def get_accuracy(self, test_data, test_labels):
        loss, accuracy = self.model.evaluate(test_data, test_labels, verbose=0)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy