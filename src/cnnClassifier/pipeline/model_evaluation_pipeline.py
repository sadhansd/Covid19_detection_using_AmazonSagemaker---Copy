from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import ModelEvaluation
from cnnClassifier import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        model_eval = ModelEvaluation()
        model_eval.evaluate_model()
        #model_eval.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e