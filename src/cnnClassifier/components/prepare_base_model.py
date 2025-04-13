import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager


class BaseModelBuilder:
    def __init__(self):
        # Initialize configuration manager and get model parameters
        config_manager = ConfigurationManager()
        config = config_manager.get_prepare_base_model_config()

        logger.info("Getting model parameters from config file")

        # Extract parameters from the configuration
        self.input_shape = config["input_shape"]
        self.parallel_kernels = config["parallel_kernels"]
        self.conv2d_filters = config["conv2d_filters"]
        self.conv2d_kernel_size = config["conv2d_kernel_size"]
        self.pool_size = config["pool_size"]
        self.dense_units = config["dense_units"]
        self.dropout_rate = config["dropout_rate"]
        self.output_units = config["output_units"]
        self.output_activation = config["output_activation"]
        self.loss = config["loss"]
        self.optimizer = config["optimizer"]
        self.metrics = config["metrics"]

    def build_model(self):
        # Input layer
        inp = Input(shape=self.input_shape)
        convs = []

        # Add parallel Conv2D layers
        for kernel_size in self.parallel_kernels:
            conv = Conv2D(self.conv2d_filters[0], kernel_size, padding='same', activation='relu', strides=1)(inp)
            convs.append(conv)

        # Concatenate the outputs of parallel Conv2D layers
        out = Concatenate()(convs)
        conv_model = Model(inp, out)

        # Build the sequential model
        model = Sequential()
        model.add(conv_model)

        model.add(Conv2D(self.conv2d_filters[1], self.conv2d_kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))

        model.add(Conv2D(self.conv2d_filters[2], self.conv2d_kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))

        model.add(Flatten())
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.dense_units[0], activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.dense_units[1], activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.output_units, activation=self.output_activation))

        # Compile the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        logger.info("Base model created successfully")
        return model


# Instantiate the BaseModelBuilder and build the model
base_model_builder = BaseModelBuilder()
model = base_model_builder.build_model()

# Print the model summary
model.summary()