import os
import cv2
import numpy as np
from cnnClassifier import logger


class DataPreprocessor:
    def __init__(self, dataset_path, img_size=100):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.categories = []
        self.labels = {}
        self.data = []
        self.target = []

    def list_categories(self):
        """List categories (directories) in the dataset path and assign labels."""
        logger.info("Starting to list directories in the dataset path.")
        label_counter = 0
        for category in os.listdir(self.dataset_path):
            if category != 'PNEUMONIA': 
                category_path = os.path.join(self.dataset_path, category)
                if os.path.isdir(category_path):  # Ensure it's a directory
                    self.categories.append(category)
                    self.labels[category] = label_counter
                    label_counter += 1
        logger.info(f"Categories found: {self.categories}")
        logger.info(f"Labels assigned: {self.labels}")

    def process_images(self):
        """Process images for each category."""
        logger.info("Starting to process images for each category.")
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            if os.path.isdir(category_path):  # Ensure it's a directory
                logger.info(f"Processing category: {category}")
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    try:
                        # Read and preprocess the image
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (self.img_size, self.img_size))
                        self.data.append(resized)
                        self.target.append(category)  # Use category as the label
                    except Exception as e:
                        logger.error(f"Exception while processing image {img_name}: {e}")

    def preprocess_data(self):
        """Convert, normalize, and reshape data."""
        logger.info("Converting data and target to numpy arrays.")
        self.data = np.array(self.data)
        self.target = np.array(self.target)

        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Target shape: {self.target.shape}")

        logger.info("Normalizing the data.")
        self.data = self.data / 255.0

        logger.info("Reshaping the data.")
        self.data = np.reshape(self.data, (self.data.shape[0], self.img_size, self.img_size, 1))

        logger.info("Mapping target categories to labels.")
        self.target = np.array([self.labels[i] for i in self.target])

    def save_data(self):
        """Save processed data and target arrays to .npy files."""
        logger.info("Saving processed data and target arrays to .npy files.")
        np.save('data', self.data)
        np.save('target', self.target)

    def load_data(self):
        """Load processed data and target arrays from .npy files."""
        logger.info("Loading processed data and target arrays from .npy files.")
        self.data = np.load('data.npy')
        self.target = np.load('target.npy')

    def get_input_shape(self):
        """Log and return the input shape for the model."""
        input_shape = self.data.shape[1:]
        logger.info(f"Input shape for the model: {input_shape}")
        return input_shape


# Example usage
if __name__ == "__main__":
    dataset_path = r'artifacts\dataset\Data\train'
    preprocessor = DataPreprocessor(dataset_path)

    preprocessor.list_categories()
    preprocessor.process_images()
    preprocessor.preprocess_data()
    preprocessor.save_data()
    preprocessor.load_data()
    input_shape = preprocessor.get_input_shape()