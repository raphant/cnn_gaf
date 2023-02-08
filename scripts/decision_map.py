from pprint import pprint

import numpy as np
from keras.backend import clear_session
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from lazyft.data_loader import load_pair_data

import util
from data import load_pair_data_for_each_timeframe
from data_models import Mode, Profile
from predict import get_single_generator
from preprocessing import (
    create_dataflow_dataframe,
    preprocess,
    preprocess2,
    preprocess_train2,
)
from split import create_decision_map

profile, model = next(util.load_specific_tuner_model("tuner/20220408181048"))
# profile = Profile(
#     pair="BTC/USDT",
#     train_timerange="20170102-20211231",
#     test_timerange="20220225-",
#     timeframes=["5m", "15m", "30m", "1h"],
#     image_size=40,
#     # download_interval='15m',
# )
profile.test_timerange = "20220225-"
# df = load_pair_data(
#     "BTC/USDT", timeframe=profile.download_interval, timerange=profile.test_timerange
# )

profile.ensure_directories_exist()
# region Non-strategy
# train_dates = data_to_image_preprocess(profile, mode=Mode.TRAIN)
# test_dates = preprocess(
#     profile, mode=Mode.TEST, load_preprocessed_file=False, save_preprocessed_file=False
# )
# dfs = load_pair_data_for_each_timeframe(profile.pair, profile.test_timerange, profile.timeframes)
test_dates = preprocess2(
    # dfs,
    profile,
    Mode.TEST,
    # profile,
)
# train_df = create_gaf_dataframe(profile.get_images_path(Mode.TRAIN), train_dates)
test_dates = sorted(test_dates)
test_df = create_dataflow_dataframe(profile.get_images_path(Mode.TEST), test_dates)
test_datagen = ImageDataGenerator(rescale=1 / 255)

target_size = (profile.image_size, profile.image_size)
class_mode = "categorical"
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=profile.get_images_path(Mode.TEST),
    target_size=target_size,
    x_col="Images",
    y_col="Labels",
    class_mode=class_mode,
    subset="training",
    shuffle=False,
)
# model: Model = load_model('tuner/20220408181048/best_model.h5')
predict = model.predict(test_generator)
# endregion
assert len(predict) == len(predict)
for i in range(len(predict)):
    assert np.array_equal(
        predict[i], predict[i]
    ), f"Arrays not equal: {predict[i]} != {predict[i]}"
clear_session()
assert len(test_dates) == len(predict)
score = model.evaluate(test_generator)
# pprint(list(zip(test_dates, [int(a[0] > 0.5) for a in predict])))
for date, prediction in zip(test_dates, predict):
    print(f"{date}: {prediction} -> {int(prediction[0] > 0.5)}")
