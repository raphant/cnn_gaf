import contextlib
import shutil
from functools import partial
from pathlib import Path

import keras
import keras_tuner as kt
from alive_progress import alive_bar
from keras.backend import clear_session
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.preprocessing.image import DirectoryIterator
from keras_preprocessing.image import ImageDataGenerator
from numpy import ndarray

from cnn_model import create_cnn_tuning, initializers, model_generator
from constants import REPO
from data_models import Mode, Profile
from preprocessing import preprocess

learning_rate_reduction = ReduceLROnPlateau(
    monitor="accuracy", patience=3, verbose=0, factor=0.5, min_lr=0.00001
)


def create_generators(
    profile: Profile,
    split: float = 0.3,
    batch_size: int = 32,
    class_mode="categorical",
    shuffle=False,
) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """
    Create generators for train, validation, and test data

    :param profile: The profile object that contains the paths to the training and testing data
    :param batch_size: The number of images to be included in each batch
    :param split: The proportion of files in the train folder to use for validation_split.
        The rest are used for training
    :param class_mode: The type of classification to use.
    :param shuffle: Whether to shuffle the data
    :return: The test_generator, train_generator, and validation_generator.
    """
    target_size = (profile.image_size, profile.image_size)
    train_validate_datagen = ImageDataGenerator(
        rescale=1 / 255, validation_split=split
    )  # set validation split
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_validate_datagen.flow_from_directory(
        directory=profile.get_images_path(Mode.TRAIN),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="training",
        shuffle=shuffle,
    )
    validation_generator = train_validate_datagen.flow_from_directory(
        directory=profile.get_images_path(Mode.TRAIN),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="validation",
        shuffle=shuffle,
    )
    test_generator = test_datagen.flow_from_directory(
        directory=profile.get_images_path(Mode.TEST),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
    )
    return train_generator, validation_generator, test_generator


def create_generators_in_memory(
    profile: Profile,
    train_data: tuple[ndarray, ndarray],
    test_data: tuple[ndarray, ndarray],
    batch_size: int = 32,
    split: float = 0.3,
):
    """
    Create generators for train, validation, and test data

    :param profile: The profile object that contains the paths to the training and testing data
    :param train_data: The training data
    :param test_data: The test data
    :param batch_size: The number of images to be included in each batch
    :param split: The proportion of files in the train folder to use for validation_split.
    :return: The test_generator, train_generator, and validation_generator.
    """
    target_size = (profile.image_size, profile.image_size)
    train_validate_datagen = ImageDataGenerator(
        rescale=1 / 255, validation_split=split
    )  # set validation split
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_validate_datagen.flow(
        *train_data, batch_size=batch_size, shuffle=True
    )
    validation_generator = train_validate_datagen.flow(
        *train_data, batch_size=batch_size, shuffle=True
    )
    test_generator = test_datagen.flow(*test_data, batch_size=batch_size, shuffle=True)
    return train_generator, validation_generator, test_generator


def main(pair: str = "BTC/USDT"):
    """
    It trains a model and saves it.

    :param pair: The pair to train on, defaults to BTC/USDT
    :type pair: str (optional)
    """

    SPLIT = 0.20
    LR = 0.001
    image_size = 40
    batch_size = 20
    EPOCHS = 10
    train_timerange = "20170102-20211231"
    test_timerange = "20220101-"
    timeframes = ["1h", "4h", "12h", "1d"]
    profile = Profile(
        pair=pair,
        train_timerange=train_timerange,
        test_timerange=test_timerange,
        timeframes=timeframes,
        image_size=image_size,
        # download_interval='15m',
    )
    profile.ensure_directories_exist()

    # cnn_networks = 1
    data_to_image_preprocess(profile, mode=Mode.TRAIN)
    data_to_image_preprocess(profile, mode=Mode.TEST)

    models = model_generator(LR, initializers, image_size)
    # train_data = get_gaf_and_answers(profile, mode=Mode.TRAIN)
    # test_data = get_gaf_and_answers(profile, mode=Mode.TEST)

    # train_generator, validation_generator, test_generator = create_generators_in_memory(
    #     profile, train_data, test_data, batch_size, SPLIT
    # )
    train_generator, validation_generator, test_generator = create_generators(
        profile, SPLIT, batch_size
    )
    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size
    with alive_bar(len(initializers), title="Training...", bar="smooth") as bar:
        for model, init in models:
            print(f"Kernel Initializer : {init.__class__.__name__}")
            train_generator.reset()
            validation_generator.reset()
            test_generator.reset()
            try:
                history = model.fit(
                    train_generator,
                    epochs=EPOCHS,
                    # steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    # validation_steps=int(validation_generator.n / valid_batch_size),
                    callbacks=[learning_rate_reduction],
                    verbose=0,
                    use_multiprocessing=True,
                )
            except Exception:
                print(model.summary())
                raise
            print(
                "CNN Model {0}: "
                "Epochs={1:d}, "
                "Training Accuracy={2:.5f}, "
                "Validation Accuracy={3:.5f}".format(
                    init,
                    EPOCHS,
                    max(history.history["acc"]),
                    max(history.history["val_acc"]),
                )
            )

            scores = model.evaluate(test_generator)
            print("Test {0}s: {1:.2f}%".format(model.metrics_names[1], scores[1] * 100))
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))
            string_list.append(f"test acc: {scores[1] * 100}")
            summary = "\n".join(string_list)
            logging = [
                "{0}: {1}".format(key, val[-1]) for key, val in history.history.items()
            ]
            log = "Results:\n" + "\n".join(logging)
            model_save_path = (
                REPO
                / pair.replace("/", "_")
                / str(profile.datestamp)
                / "models"
                / f"{init.__class__.__name__}.h5"
            )
            summary_save_path = (
                REPO
                / pair.replace("/", "_")
                / str(profile.datestamp)
                / "summaries"
                / f"{init.__class__.__name__}-{scores[1] * 100:.2f}.txt"
            )

            model_save_path.parent.mkdir(exist_ok=True, parents=True)
            summary_save_path.parent.mkdir(exist_ok=True, parents=True)

            model.save(model_save_path)
            profile.save()
            summary_save_path.write_text(
                f"EPOCHS: {EPOCHS}\nSteps per epoch: {steps_per_epoch}\n"
                f"Validation steps: {validation_steps}\n"
                f"Val Split:{SPLIT}\nLearning RT:{summary}\n\n\n{LR}"
                f"\n\n=========TRAINING LOG========\n{log}"
            )
            bar()
            clear_session()
            del model


def main_auto_ml():
    train_timerange = "20170102-20211231"
    test_timerange = "20220101-"
    timeframes = ["1h", "4h", "12h", "1d"]
    profile = Profile(
        pair="BTC/USDT",
        train_timerange="20170102-20211231",
        test_timerange="20220101-",
        download_interval="1h",
        timeframes=["1h", "4h", "12h", "1d"],
        image_size=40,
        # download_interval='15m',
    )
    build_model = partial(
        create_cnn_tuning,
        image_size=profile.image_size,
    )
    # data_to_image_preprocess(profile, mode=Mode.TRAIN)
    # data_to_image_preprocess(profile, mode=Mode.TEST)

    train_generator, validation_generator, test_generator = create_generators(
        profile, 0.0, batch_size=3154
    )
    train_img, train_labels = train_generator.next()
    test_img, test_labels = test_generator.next()
    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=30,
        overwrite=True,
        directory="tuner",
        project_name=profile.datestamp,
        executions_per_trial=2,
    )

    tuner.search(
        train_img,
        train_labels,
        validation_data=(test_img, test_labels),
        batch_size=32,
        epochs=25,
        callbacks=[keras.callbacks.TensorBoard(f"tuner/tb{profile.datestamp}")],
    )

    print(tuner.results_summary())
    print(tuner.get_best_hyperparameters()[0].values)
    models = tuner.get_best_models()

    best_model = models[0]

    scores = best_model.evaluate(test_img, test_labels, return_dict=True)
    if scores["accuracy"] > 0.6:
        best_model.save(Path("tuner", profile.datestamp, "best_model.h5"))
        profile.save(directory=Path("tuner", profile.datestamp))
        best_model.summary()
    else:
        shutil.rmtree(Path("tuner", profile.datestamp))


def evaluate_model(model_path: Path):
    model: Sequential = load_model(model_path)
    profile = Profile(
        pair="BTC/USDT",
        train_timerange="20170102-20211231",
        test_timerange="20220101-",
        download_interval="1h",
        timeframes=["1h", "4h", "12h", "1d"],
        image_size=40,
    )
    train_generator, validation_generator, test_generator = create_generators(
        profile, 0.3, batch_size=32
    )
    train_img, train_labels = train_generator.next()
    test_img, test_labels = test_generator.next()
    history = model.fit(
        train_generator,
        epochs=5,
        batch_size=32,
        # steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        # validation_steps=int(validation_generator.n / valid_batch_size),
        callbacks=[learning_rate_reduction],
        # verbose=0,
        use_multiprocessing=True,
    )
    print(history.history)
    with contextlib.suppress(Exception):
        print(
            "CNN Model {0}: "
            "Epochs={1:d}, "
            "Training Accuracy={2:.5f}, "
            "Validation Accuracy={3:.5f}".format(
                "tuner",
                100,
                max(history.history["acc"]),
                max(history.history["val_acc"]),
            )
        )
    score = model.evaluate(test_generator, return_dict=True)
    if score["accuracy"] > 0.65:
        model.save(Path(model_path, "trained.h5"))
        # profile.save(directory=model_path.parent)
        model.summary()
    # print(scores)
    # print("Test {0}s: {1:.2f}%".format(model.metrics_names[1], scores[1] * 100))


if __name__ == "__main__":
    main("BTC/USDT")
    # evaluate_model(model_path=Path('tuner/20220406161500/best_model.h5'))
    # main_auto_ml()
