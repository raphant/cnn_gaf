import keras
import keras_tuner as kt
from keras import layers
from keras.backend import categorical_crossentropy
from keras.optimizer_v2.adam import Adam


class MyHyperModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters):
        conv1_units = hp.Int('conv1_units', min_value=32, max_value=128, step=16)
        conv2_units = hp.Int('conv2_units', min_value=32, max_value=128, step=16)
        conv3_units = hp.Int('conv3_units', min_value=32, max_value=128, step=16)
        conv4_units = hp.Int('conv4_units', min_value=32, max_value=128, step=16)
        conv5_units = hp.Int('conv5_units', min_value=32, max_value=128, step=16)
        dense1_units = hp.Int('dense1_units', min_value=256, max_value=2048, step=128)
        conv1_kernel = hp.Choice('conv1_kernel', [3, 5])
        learning_rate = hp.Choice('learning_rate', [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
        epsilon = hp.Choice('epsilon', [1e-8, 1e-6, 1e-4])
        decay = hp.Choice('decay', [1e-6, 1e-4, 1e-2])
        # beta_1 = hp.Choice('beta_1', [0.1, 0.2, 0.3, 0.4, 0.5])
        # beta_2 = hp.Choice('beta_2', [0.1, 0.2, 0.3, 0.4, 0.5])
        model = keras.Sequential(
            [
                #  First Convolution
                keras.layers.Conv2D(
                    conv1_units,
                    kernel_size=conv1_kernel,
                    activation='relu',
                    input_shape=(image_size, image_size, 3),
                ),
                # keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
                keras.layers.MaxPooling2D(pool_size=2, strides=2),
                keras.layers.Dropout(0.25),
                # Second Convolution
                keras.layers.Conv2D(conv2_units, kernel_size=3, activation='relu', padding='same'),
                keras.layers.Conv2D(conv3_units, kernel_size=3, activation='relu', padding='same'),
                keras.layers.MaxPooling2D(pool_size=2, strides=2),
                keras.layers.Dropout(0.25),
                # Third Convolution
                keras.layers.Conv2D(conv4_units, kernel_size=3, activation='relu', padding='same'),
                keras.layers.Conv2D(conv5_units, kernel_size=3, activation='relu', padding='same'),
                keras.layers.MaxPooling2D(pool_size=2, strides=2),
                # Output layer
                keras.layers.Flatten(),
                keras.layers.Dense(dense1_units, activation='relu'),
                keras.layers.Dense(2, activation='softmax'),
            ]
        )
        model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon),
            metrics=['accuracy'],
        )
        return model

    # noinspection PyMethodOverriding
    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        return model.fit(
            x,
            y,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )
