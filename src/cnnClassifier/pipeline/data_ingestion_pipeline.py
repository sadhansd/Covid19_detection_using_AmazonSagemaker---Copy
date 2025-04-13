from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import *
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        data_path,output_dir = self.config.get_data_ingestion_config()
        ingestor = dataIngestorFactory.get_data_ingestor(data_path, output_dir)
        extracted_path = ingestor.get_data()
        logger.info(f"Data ingested and available at path: /{extracted_path}")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e