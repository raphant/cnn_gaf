import datetime as dt
import enum
import math
from pathlib import Path
from typing import Optional

from freqtrade.exchange import timeframe_to_minutes
from keras import Sequential
from keras.models import load_model
from loguru import logger
from pydantic import BaseModel, Field

import util
from constants import DEFAULT_TIMEFRAMES, IMAGES_PATH, REPO


class Mode(enum.Enum):
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"
    LIVE = "live"


class Profile(BaseModel):
    """
    Profile class for training and testing
    """

    pair: str = Field(..., title="Pair", description="The pair of currencies to trade")
    train_timerange: str = Field(..., description="Training timerange")
    test_timerange: Optional[str] = Field("", description="The timerange for the test images")
    window_size = Field(20, description="The size of the window to use in days")
    image_size = Field(40, description="What size the images should be loaded as")
    batch_size: int = Field(32, description="The batch size to feel the network")
    tail_window: int = Field(20, description="The number of candles to tail for each timeframe")

    timeframes: list[str] = Field(
        default_factory=lambda: DEFAULT_TIMEFRAMES, description="The timeframes to use"
    )
    datestamp: str = Field(
        default_factory=lambda: dt.datetime.now().strftime("%Y%m%d%H%M%S"),
        description="The date of the run",
    )

    @property
    def formatted_pair(self):
        """
        Returns the pair with '/' replaced by '_'

        # Python
        def formatted_pair(self):
                return self.pair.replace('/', '_')
        :return: A string of the pair formatted for use in a filename.
        """
        return self.pair.replace("/", "_")

    @property
    def download_interval(self):
        return self.min_tf

    @property
    def max_tf(self):
        """
        Return the maximum value in the list of timeframes
        :return: The maximum timeframe in the list of timeframes.
        """
        return max(self.timeframes, key=lambda tf: timeframe_to_minutes(tf))

    @property
    def min_tf(self):
        """
        Return the smallest element in a list
        :return: The minimum timeframe in the list of timeframes.
        """
        return min(self.timeframes, key=lambda tf: timeframe_to_minutes(tf))

    @property
    def step_size(self):
        return math.ceil(
            self.window_size / (timeframe_to_minutes(self.download_interval) / 60) * 24
        )

    @property
    def max_tf_as_hours(self):
        """
        Return the maximum value in the list of timeframes as hours

        :return: The maximum timeframe in the list of timeframes.
        """
        return timeframe_to_minutes(self.max_tf) // 60

    @property
    def model_save_path(self):
        """
        The function returns the path to the model file
        :return: A path to the model file.
        """
        path = REPO.joinpath(self.formatted_pair, self.datestamp)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def preprocess_directory(self):
        return

    def candles_needed(self, days: int) -> int:
        MINUTES_IN_DAY = 1440
        tf_minutes = timeframe_to_minutes(self.max_tf)
        return days * MINUTES_IN_DAY // tf_minutes

    def save(self, directory: Path = None):
        """
        The function takes the current instance of the class and saves it to a file
        """
        directory = directory or self.model_save_path
        path = directory.joinpath("profile.json")
        if path.exists():
            return
        path.write_text(self.json())
        logger.info(f"Saved profile to {path}")

    def get_timerange(self, mode: Mode):
        """
        The function returns the timerange for a given mode

        :param mode: The mode of the images. Either train or test
        :return: A string of the timerange.
        """
        if mode == Mode.TRAIN:
            return self.train_timerange
        elif mode == Mode.TEST:
            return self.test_timerange
        else:
            raise ValueError(f"Mode {mode} is not supported")

    def get_images_path(self, mode: Mode):
        """
        The function returns the path to the images for a given pair of currencies and a given
        timerange
        :param mode: The mode of the images. Either train or test
        :return: A path to the images.
        """
        if mode == Mode.PREDICT:
            return IMAGES_PATH.joinpath(self.formatted_pair, str(self.tail_window), "predict")
        return (
            IMAGES_PATH
            / self.formatted_pair
            / self.get_timerange(mode)
            / str(self.tail_window)
            / "_".join(self.timeframes)
        )

    def ensure_directories_exist(self):
        """
        Make sure that the directories for the images exist
        """
        for mode in [Mode.TRAIN, Mode.TEST, Mode.PREDICT]:
            self.get_images_path(mode).mkdir(parents=True, exist_ok=True)

    def available_models(self):
        """
        Return a list of all the models in the model_save_path directory

        :return: A list of the available models.
        """
        return list(self.model_save_path.joinpath("models").glob("*.h5"))

    @util.timeit
    def get_model(self, model_name: str) -> Sequential:
        """
        Load a model from a file

        :param model_name: The name of the model to be loaded
        :return: The model itself.
        """
        return load_model(self.model_save_path.joinpath("models", model_name))

    @classmethod
    def load(cls, pair: str, datestamp: str) -> "Profile":
        """
        The function takes a pair and a datestamp and returns the profile
        """
        return Profile.parse_file(REPO.joinpath(pair.replace("/", "_"), datestamp, "profile.json"))
