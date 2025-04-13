from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preprocessing import DataPreprocessor
from cnnClassifier import logger



STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        dataset_path = self.config.get_data_preprocessing_config()
        self.preprocessor = DataPreprocessor(dataset_path)
        self.preprocessor.list_categories()
        self.preprocessor.process_images()
        self.preprocessor.preprocess_data()
        self.preprocessor.save_data()
        self.preprocessor.load_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e