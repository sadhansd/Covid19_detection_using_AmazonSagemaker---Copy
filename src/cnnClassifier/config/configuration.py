import confuse
from cnnClassifier import logger

class ConfigurationManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = config_path
    
    def get_data_ingestion_config(self):
        config = confuse.Configuration("config", __name__)
        config.set_file(self.config)
        dataset_name = config["dataset"]["dataset_name"].get(str)
        output_dir = config["dataset"]["output_dir"].get(str)
        logger.info("got dataset details")
        return dataset_name, output_dir
    
    def get_data_preprocessing_config(self):
        config = confuse.Configuration("config", __name__)
        config.set_file(self.config)
        dataset_path = config["training"]["data_path"].get(str)
        return dataset_path


    def get_prepare_base_model_config(self):
        config = confuse.Configuration("Covid19Detection", __name__)
        config.set_file(self.config)
        input_shape = tuple(config["model"]["input_shape"].get(list))
        parallel_kernels = config["model"]["parallel_kernels"].get(list)
        conv2d_filters = config["model"]["conv2d_filters"].get(list)
        conv2d_kernel_size = tuple(config["model"]["conv2d_kernel_size"].get(list))
        pool_size = tuple(config["model"]["pool_size"].get(list))
        dense_units = config["model"]["dense_units"].get(list)
        dropout_rate = config["model"]["dropout_rate"].get(float)
        output_units = config["model"]["output_units"].get(int)
        output_activation = config["model"]["output_activation"].get(str)
        loss = config["training"]["loss"].get(str)
        optimizer = config["training"]["optimizer"].get(str)
        metrics = config["training"]["metrics"].get(list)

        return {
            "input_shape": input_shape,
            "parallel_kernels": parallel_kernels,
            "conv2d_filters": conv2d_filters,
            "conv2d_kernel_size": conv2d_kernel_size,
            "pool_size": pool_size,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "output_units": output_units,
            "output_activation": output_activation,
            "loss": loss,
            "optimizer": optimizer,
            "metrics": metrics
        }