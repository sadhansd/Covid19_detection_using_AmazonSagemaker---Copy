from cnnClassifier.components.prepare_base_model import BaseModelBuilder
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import numpy as np

class ModelTraining:
    def __init__(self):
        self.config = ConfigurationManager()
        self.model_builder = BaseModelBuilder()

    def split_data(self):
        feature_path, target_path, test_size, = self.config.get_data_splitting_config()
        data= np.load(feature_path)
        target= np.load(target_path)
        target = to_categorical(target)
        logger.info(target.shape)

        train_data, test_data, train_target, test_target = train_test_split(
            data,target, test_size=test_size
        )
        logger.info("Splitting data into training and testing sets")
        return train_data, test_data, train_target, test_target

    def set_checkpoint(self):
        checkpoint = ModelCheckpoint('model-{epoch:03d}.keras',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
        return checkpoint
    
    def train_model(self):

        self.model = self.model_builder.build_model()
        self.epoch, self.validation_split, self.model_save_path = self.config.get_model_training_config()
        logger.info("Model training configuration loaded successfully")
        train_data, test_data, train_target, test_target = self.split_data()
        checkpoint = self.set_checkpoint()
        logger.info("Training the model")
        history=self.model.fit(train_data,train_target,epochs=self.epoch,callbacks=[checkpoint],validation_split=self.validation_split,verbose=1)
        logger.info("Model training completed")
        self.model.save(self.model_save_path)
        logger.info("Model saved successfully")

if __name__ == "__main__":
    model_training = ModelTraining()
    model_training.train_model()