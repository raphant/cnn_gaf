import keras
import keras_tuner
from keras import losses
from keras.initializers import initializers_v2 as initializer
from keras.initializers.initializers_v2 import Initializer
from keras.layers import BatchNormalization, Conv2D, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizer_v2.adam import Adam

#  Ensemble CNN network to train a CNN model on GAF images labeled Long and Short

initializers = {
    "none": None,
    "Orthogonal": initializer.Orthogonal(),
    "LecunUniform": initializer.LecunUniform(),
    "VarianceScaling": initializer.VarianceScaling(),
    "RandomNormal": initializer.RandomNormal(),
    "RandomUniform": initializer.RandomUniform(),
    "TruncatedNormal": initializer.TruncatedNormal(),
    "GlorotNormal": initializer.GlorotNormal(),
    "GlorotUniform": initializer.GlorotUniform(),
    "HeNormal": initializer.HeNormal(),
    "HeUniform": initializer.HeUniform(),
    # 'Orthogonal2': initializer.Orthogonal(seed=42),
    # 'LecunUniform2': initializer.LecunUniform(seed=42),
    # 'VarianceScaling2': initializer.VarianceScaling(seed=42),
    # 'RandomNormal2': initializer.RandomNormal(seed=42),
    # 'RandomUniform2': initializer.RandomUniform(seed=42),
    # 'TruncatedNormal2': initializer.TruncatedNormal(seed=42),
    # 'GlorotNormal2': initializer.GlorotNormal(seed=42),
    # 'GlorotUniform2': initializer.GlorotUniform(seed=42),
    # 'HeNormal2': initializer.HeNormal(seed=42),
    # 'HeUniform2': initializer.HeUniform(seed=42),
}


def create_cnn(image_size: int, kernel_initializer=None) -> keras.Sequential:
    """
    Create a CNN with 3 convolutional layers and 2 dense layers

    :param image_size: The size of the image data
    :type image_size: int
    :param kernel_initializer: The kernel initializer for the convolutional layers
    :return: A neural network model.
    """
    return keras.Sequential(
        [
            #  First Convolution
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                activation="relu",
                input_shape=(image_size, image_size, 3),
                padding="same",
                kernel_initializer=kernel_initializer,
            ),
            # keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
            # keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Dropout(0.25),
            # Second Convolution
            keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            # keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Dropout(0.25),
            # Third Convolution
            keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
            keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
            # keras.layers.MaxPooling2D(pool_size=2, strides=2),
            # Output layer
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )


def create_cnn_tuning(
    hp: keras_tuner.HyperParameters, image_size: int, kernel_initializer=None
) -> keras.Sequential:
    """
    Create a CNN with 3 convolutional layers and 2 dense layers

    :param hp: The hyperparameters for the model
    :param image_size: The size of the image data
    :type image_size: int
    :param kernel_initializer: The kernel initializer for the convolutional layers
    :return: A neural network model.
    """

    # conv1_units = hp.Int('conv1_units', min_value=32, max_value=128, step=16)
    # conv1_kernel = hp.Choice('conv1_kernel', [3, 5, 9])
    # conv2_units = hp.Int('conv2_units', min_value=32, max_value=128, step=16)
    # conv2_kernel = hp.Choice('conv2_kernel', [3, 5, 9])
    # conv3_units = hp.Int('conv3_units', min_value=32, max_value=128, step=16)
    # conv3_kernel = hp.Choice('conv3_kernel', [3, 5, 9])
    # conv4_units = hp.Int('conv4_units', min_value=32, max_value=128, step=16)
    # conv4_kernel = hp.Choice('conv4_kernel', [3, 5, 9])
    # conv5_units = hp.Int('conv5_units', min_value=32, max_value=128, step=16)
    # conv5_kernel = hp.Choice('conv5_kernel', [3, 5, 9])
    # conv6_units = hp.Int('conv6_units', min_value=32, max_value=128, step=16)
    # conv6_kernel = hp.Choice('conv6_kernel', [3, 5, 9])
    conv_dict = {}
    for i in range(1, 7 + 1):
        conv_dict[f"conv{i}_units"] = hp.Int(
            f"conv{i}_units", min_value=32, max_value=128, step=16
        )

        conv_dict[f"conv{i}_kernel"] = hp.Choice(f"conv{i}_kernel", [3, 5])

    # dense1_units = hp.Int('dense1_units', min_value=256, max_value=2048, step=128)
    dropout1 = hp.Float(
        "dropout1", min_value=0.0, max_value=0.5, step=0.05, default=0.4
    )
    dropout2 = hp.Float(
        "dropout2", min_value=0.0, max_value=0.5, step=0.05, default=0.4
    )
    loss_dict = {
        "mean_absolute_error": losses.mean_absolute_error,
        "mean_absolute_percentage_error": losses.mean_absolute_percentage_error,
        "categorical_crossentropy": losses.categorical_crossentropy,
        "mean_squared_error": losses.mean_squared_error,
    }
    loss_function = hp.Choice(
        "loss",
        list(loss_dict.keys()),
    )
    # learning_rate = hp.Choice('learning_rate', [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
    # epsilon = hp.Choice('epsilon', [1e-8, 1e-6, 1e-4])
    # decay = hp.Choice('decay', [1e-6, 1e-4, 1e-2])
    # beta1 = hp.Choice('beta1', [0.9, 0.95, 0.99])
    # beta2 = hp.Choice('beta2', [0.9, 0.95, 0.99])
    # beta_1 = hp.Choice('beta_1', [0.1, 0.2, 0.3, 0.4, 0.5])
    # beta_2 = hp.Choice('beta_2', [0.1, 0.2, 0.3, 0.4, 0.5])
    model = keras.Sequential(
        [
            # first convolution
            Conv2D(
                conv_dict["conv1_units"],
                kernel_size=conv_dict["conv1_kernel"],
                activation="relu",
                input_shape=(image_size, image_size, 3),
                kernel_initializer=kernel_initializer,
            ),
            BatchNormalization(),
            Conv2D(
                conv_dict["conv2_units"],
                kernel_size=conv_dict["conv2_kernel"],
                activation="relu",
            ),
            BatchNormalization(),
            Conv2D(
                conv_dict["conv3_units"],
                kernel_size=conv_dict["conv3_kernel"],
                strides=2,
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            Dropout(dropout1),
            # Second Convolution
            Conv2D(
                conv_dict["conv4_units"], conv_dict["conv4_kernel"], activation="relu"
            ),
            BatchNormalization(),
            Conv2D(
                conv_dict["conv5_units"],
                kernel_size=conv_dict["conv5_kernel"],
                activation="relu",
            ),
            BatchNormalization(),
            Conv2D(
                conv_dict["conv6_units"],
                kernel_size=conv_dict["conv6_kernel"],
                strides=2,
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            Dropout(dropout1),
            # Third Convolution
            Conv2D(
                conv_dict["conv7_units"],
                kernel_size=conv_dict["conv7_kernel"],
                activation="relu",
            ),
            BatchNormalization(),
            Flatten(),
            Dropout(dropout2),
            # Output layer
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        loss=loss_dict[loss_function],
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def create_cnn_alternative(
    image_size: int, kernel_initializer=None
) -> keras.Sequential:
    """
    Create a CNN with 3 convolutional layers, each with a dropout layer after it

    :param image_size: The size of the image
    :type image_size: int
    :param kernel_initializer: The kernel initializer for the convolutional layers
    :return: A compiled model.
    """
    return keras.Sequential(
        [
            #  First Convolution
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(image_size, image_size, 3),
                kernel_initializer=kernel_initializer,
            ),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(
                32, kernel_size=(3, 3), strides=2, padding="same", activation="relu"
            ),
            BatchNormalization(),
            Dropout(0.4),
            # Second Convolution
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(
                64, kernel_size=(3, 3), strides=2, padding="same", activation="relu"
            ),
            BatchNormalization(),
            Dropout(0.4),
            # Third Convolution
            Conv2D(128, kernel_size=4, activation="relu"),
            BatchNormalization(),
            Flatten(),
            Dropout(0.4),
            # Output layer
            keras.layers.Dense(2, activation="softmax"),
        ]
    )


def model_generator(
    learning_rate: float, initializers: dict, target_size: int
) -> tuple[keras.Sequential, Initializer]:
    """
    Create a CNN model with a given kernel initializer and compile it with Adam optimizer and
    categorical crossentropy loss

    :param learning_rate: The learning rate for the model
    :param initializers: a dictionary of initializers
    :param target_size: The size of the images that will be fed to the CNN
    """
    for i in initializers.values():
        cnn = create_cnn_alternative(target_size, kernel_initializer=i)
        # Compile each model
        cnn.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=categorical_crossentropy,
            metrics=["acc"],
        )
        yield cnn, i
