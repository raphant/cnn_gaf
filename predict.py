from pathlib import Path

import keras
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from lazyft.data_loader import load_pair_data

from cnn_model import create_cnn
from constants import REPO
from data_models import Mode, Profile
from preprocessing import create_dataflow_dataframe


def predict(
    data: pd.DataFrame = None, model: keras.Sequential = None, pair="BTC/USDT"
) -> int:
    if data is None:
        data = load_pair_data(pair, "1h", timerange="20220312-")
    images = quick_gaf(data)
    # train_dataset = datasets.ImageFolder(
    #     root=str(tmp),
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(255),
    #             transforms.ToTensor()
    #             # transforms.Scale(255),
    #         ]
    #     ),
    # )
    if not model:
        model = create_cnn(40)
        model_to_load = (
            REPO / "models" / pair.replace("/", "_") / "20220403011222_GlorotUniform.h5"
        )
        try:
            model.load_weights(model_to_load)
        except OSError as e:
            raise OSError(f"Could not load model {model_to_load}") from e
    # x = np.resize(preprocessed[0], 255)

    x = transform(images)
    prediction = model.predict(x[-1])
    print(prediction, type(prediction))
    return prediction[0][0]


def get_single_generator(
    profile: Profile,
    class_mode="categorical",
    folder: Path = None,
    dates: list[str] = None,
):
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    test_df = create_dataflow_dataframe(folder, dates)
    return test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=folder,
        batch_size=len(dates),
        target_size=(profile.image_size, profile.image_size),
        x_col="Images",
        y_col="Labels",
        class_mode=class_mode,
        subset="training",
        shuffle=False,
    )
