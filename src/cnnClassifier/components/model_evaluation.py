from tensorflow.keras.models import load_model
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier.utils.common import save_json
from pathlib import Path
import mlflow
from urllib.parse import urlparse
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from mlflow.models.signature import infer_signature



class ModelEvaluation:

    def __init__(self):
        self.config = ConfigurationManager()
        self.mlflow_uri = self.config.get_mlflow_uri()
        self.model_path = self.config.get_model_path()

    def load_model(self):
        model = load_model(self.model_path)
        self.model = model
        return model
    
    def evaluate_model(self):
        model = self.load_model()
        obj = ModelTraining()
        train_data, test_data, train_target, test_target = obj.split_data()
        self.score = model.evaluate(test_data,test_target)
        self.save_score()
    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.mlflow_uri)

        import os

        os.environ["MLFLOW_TRACKING_USERNAME"] = "sadhansd"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "e0c4183d9bcd30f40adc164109c5bb3dee8caa03"
        logger.info(f"MLflow tracking URI: {self.mlflow_uri}")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        logger.info(f"{self.config.get_epoch()}")

        with mlflow.start_run():
            mlflow.log_params({"epoch": self.config.get_epoch()})
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="Covid19DetectionModel"
                )
            else:
                mlflow.keras.log_model(
                    self.model,
                    "model" )
