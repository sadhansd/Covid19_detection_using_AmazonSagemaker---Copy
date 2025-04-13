from abc import ABC, abstractmethod
import zipfile
import os
import pandas as pd
import subprocess
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

class Ingestor(ABC):
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir

    @abstractmethod
    def get_data(self):
        pass

class zipIngestor(Ingestor):
    def get_data(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
        
        extracted_files = os.listdir(self.output_dir)
        csv_files = [file for file in extracted_files if file.endswith('.csv')]
        
        if len(csv_files) == 0:
            logger.error("No CSV files found in the zip archive.")
            raise ValueError("No CSV files found in the zip archive.")
        
        logger.info(f"Data extracted to {self.output_dir}")
        return self.output_dir

class csvIngestor(Ingestor):
    def get_data(self):
        os.makedirs(self.output_dir, exist_ok=True)
        output_file_path = os.path.join(self.output_dir, os.path.basename(self.data_path))
        pd.read_csv(self.data_path).to_csv(output_file_path, index=False)
        logger.info(f"CSV file copied to {output_file_path}")
        return self.output_dir

class kaggleIngestor(Ingestor):
    def get_data(self):
        # Ensure the Kaggle API is installed
        try:
            subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            logger.error("Kaggle API is not installed. Install it using 'pip install kaggle'.")
            raise EnvironmentError("Kaggle API is not installed. Install it using 'pip install kaggle'.")

        os.makedirs(self.output_dir, exist_ok=True)

        # Download the dataset
        dataset_name = self.data_path  # Here, `data_path` is the Kaggle dataset identifier
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", self.output_dir], check=True)
            logger.info(f"Dataset '{dataset_name}' downloaded successfully to '{self.output_dir}'.")
        except Exception as e:
            logger.error(f"Error downloading dataset from Kaggle: {e}")
            raise RuntimeError(f"Error downloading dataset from Kaggle: {e}")

        # Find and extract the downloaded zip file
        downloaded_files = os.listdir(self.output_dir)
        zip_files = [file for file in downloaded_files if file.endswith('.zip')]

        if len(zip_files) == 0:
            logger.error("No zip files found in the downloaded Kaggle dataset.")
            raise ValueError("No zip files found in the downloaded Kaggle dataset.")

        zip_file_path = os.path.join(self.output_dir, zip_files[0])
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)

        logger.info(f"Data extracted to {self.output_dir}")
        return self.output_dir

class dataIngestorFactory:
    @staticmethod
    def get_data_ingestor(data_path, output_dir):
        if data_path.endswith('.zip'):
            return zipIngestor(data_path, output_dir)
        elif data_path.endswith('.csv'):
            return csvIngestor(data_path, output_dir)
        elif "/" in data_path:  # Assuming Kaggle dataset identifiers contain slashes
            return kaggleIngestor(data_path, output_dir)
        else:
            logger.error(f"No ingestor available for {data_path}")
            raise ValueError(f'No ingestor available for {data_path}')

class Ingest_data:
    def __init__(self):
        self.config = ConfigurationManager()
        data_path,output_dir = self.config.get_data_ingestion_config()
        ingestor = dataIngestorFactory.get_data_ingestor(data_path, output_dir)
        extracted_path = ingestor.get_data()
        logger.info(f"Data ingested and available at path: /{extracted_path}")