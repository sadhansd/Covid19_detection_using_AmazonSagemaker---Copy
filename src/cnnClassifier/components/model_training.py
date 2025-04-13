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
        self.model = self.model_builder.build_model()
        self.model_path, self.feature_data, self.target_data, self.test_size, self.epoch, self.validation_split, self.model_save_path = self.config.get_model_training_config()
        logger.info("Getting model parameters from config file")
        logger.info("Loading feature and target data")

    def split_data(self):

        data= np.load(self.feature_data)
        target= np.load(self.target_data)
        target = to_categorical(target)
        logger.info(target.shape)

        train_data, test_data, train_target, test_target = train_test_split(
            data,target, test_size=self.test_size
        )
        logger.info("Splitting data into training and testing sets")
        return train_data, test_data, train_target, test_target

    def set_checkpoint(self):
        checkpoint = ModelCheckpoint('model-{epoch:03d}.keras',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
        return checkpoint
    
    def train_model(self):
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