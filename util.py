import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Union

import numpy as np
import pandas as pd
from keras.models import load_model
from numpy.lib.stride_tricks import as_strided as stride

from cnn_model import create_cnn

if TYPE_CHECKING:
    from data_models import Profile


def print_preprocess_info(decision_map, dt_points):
    total_shorts = len(decision_map["SHORT"])
    total_longs = len(decision_map["LONG"])
    images_created = total_shorts + total_longs
    print(
        "========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}"
        "\nTotal LONG positions: {2}\nTotal SHORT positions: {3}".format(
            dt_points, images_created, total_shorts, total_longs
        )
    )


def get_dates_from_df(df: pd.DataFrame, interval: str) -> list[str]:
    """
    Given a dataframe and an interval, return a list of dates that are present in the dataframe

    :param df: The dataframe to be grouped
    :param interval: The time interval to group the data
    :return: A list of dates.
    """
    grouped = group_by(df, convert_timeframe_to_grouper_compatible(interval), dropna=False)
    return grouped["date"].drop_duplicates().tolist()


def hash(obj):
    """
    Since hash() is not guaranteed to give the same result in different
    sessions, we will be using hashlib for more consistent hash_ids
    """
    if isinstance(obj, (set, tuple, list, dict)):
        obj = repr(obj)
    hash_id = hashlib.md5()
    hash_id.update(repr(obj).encode('utf-8'))
    hex_digest = str(hash_id.hexdigest())
    return hex_digest


def group_by(df: pd.DataFrame, interval: str, key="date", dropna=True) -> pd.DataFrame:
    """
    Given a dataframe, group it by a given interval and return the mean of the grouped dataframe

    :param df: The dataframe to be grouped
    :param interval: The frequency of the groupby
    :param key: The name of the column to group by, defaults to date (optional)
    :param dropna: Whether or not to drop the NaN values, defaults to True (optional)
    :return: A dataframe with the mean of the values in the dataframe grouped by the interval.
    """
    index = df.groupby(pd.Grouper(key=key, freq=interval)).mean().reset_index()
    if dropna:
        index = index.dropna()

    return index


def enumerate_step(iterable: Iterable, start=0, step=1):
    """
    Given a list, return a generator that yields the index of each element in the list and the element
    itself

    :param iterable: The list of items to iterate over
    :param start: the starting value of the sequence, defaults to 0 (optional)
    :param step: the amount to increment the counter by each time, defaults to 1 (optional)
    """
    index = start
    for x in iterable[start::step]:
        yield index, x
        index += step


def timeit(func):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' executed in {end - start:.4f}s")
        return result

    return wrapped


def roll(df, w, **kwargs):
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat(
        {row: pd.DataFrame(values, columns=df.columns) for row, values in zip(df.index, a)}
    )

    # return rolled_df.groupby(np.arange(len(df)) // w, level=0, **kwargs)
    return rolled_df.groupby(level=0, **kwargs)


def batch_iterator(data: pd.DataFrame, window: int):
    """
    It takes a dataframe and a window size, and returns a batch iterator

    :param data: the dataframe containing the data
    :type data: pd.DataFrame
    :param window: the number of days to look back
    :type window: int
    :return: A groupby object
    """
    return data.groupby(np.arange(len(data)) // window)


def load_tuner_models(n=1000) -> tuple["Profile", Path]:
    """
    It loads all the models in the tuner directory
    """
    from data_models import Profile

    directory = Path("tuner")
    # get all h5 files using glob
    models = list(directory.glob("*/*.h5"))
    profiles = [Profile.parse_file(f.with_name("profile.json")) for f in models]
    zipped = list(zip(profiles, models))[:n]
    for p, f in zipped:
        yield p, load_model(f)


def load_specific_tuner_model(directory: Union[str, Path]):
    """
    It loads the best model and profile from a directory

    :param directory: The directory where the model is stored
    :type directory: Union[str, Path]
    """
    from data_models import Profile

    if not isinstance(directory, Path):
        directory = Path(directory)
    model = load_model(directory / "best_model.h5")
    profile = Profile.parse_file(directory / "profile.json")
    yield profile, model


def load_model_from_directory(directory: Union[str, Path]):
    from data_models import Profile

    if isinstance(directory, str):
        directory = Path(directory)
    profile = Profile.parse_file(directory / "profile.json")
    model = load_model(directory)
    return [(profile, model)]


def load_fake_model():
    from data_models import Profile

    model = create_cnn(40)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return [Profile(pair="dummy", train_timerange=""), model]


def convert_timeframe_to_grouper_compatible(timeframe: str):
    return timeframe.replace("m", "min")


if __name__ == "__main__":
    # print(load_model_from_directory("90pct"))
    print(convert_timeframe_to_grouper_compatible("5m"))


def get_dates_from_df2(df: pd.DataFrame):
    return df["date"].drop_duplicates().tolist()
