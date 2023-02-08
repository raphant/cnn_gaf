# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from typing import Optional

import numpy as np  # noqa
import pandas as pd  # noqa
from freqtrade.enums import SellType
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy, SellCheckTuple
from keras import Sequential
from keras.backend import clear_session
from keras.models import Model, load_model
from keras_preprocessing.image import ImageDataGenerator
from lazyft.space_handler import SpaceHandler
from pandas import DataFrame

from constants import REPO
from data_models import Mode, Profile
from predict import get_single_generator
from preprocessing import create_dataflow_dataframe, preprocess, preprocess_train2
from util import load_specific_tuner_model

logger = logging.getLogger(__name__)

COLUMNS_FILTER = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "buy",
    "sell",
    "volume",
    "buy_tag",
    "exit_tag",
]
from scipy import stats

stats.zscore = partial(stats.zscore, nan_policy="omit")


class SagesGymCNN2(IStrategy):
    sh = SpaceHandler(__file__)

    # # If you've used SimpleROIEnv then use this minimal_roi
    # minimal_roi = {
    #     "720": -10,
    #     "600": 0.00001,
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.03
    # }

    minimal_roi = {"0": 100}

    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.017
    trailing_only_offset_is_reached = True

    timeframe = "1h"

    use_sell_signal = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    startup_candle_count: int = 960
    # ignore_roi_if_buy_signal = True
    step_size = None
    models: dict[str, list[tuple[Profile, str]]] = defaultdict(list)
    models_: list[tuple[Profile, Model]] = []
    specific_model = sh.get_setting("specific_model", None)
    padding = 0
    min_buy_threshold = sh.get_setting("min_buy_threshold", 0.6)
    max_tf = '1d'
    order_types = {
        "buy": "limit",
        "sell": "limit",
        "emergencysell": "market",
        "forcebuy": "market",
        "forcesell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if self.specific_model:
            logger.info(f"Using specific model: {self.specific_model}")
            self.models_ = load_specific_tuner_model(self.specific_model)
            # self.models_ = load_model_from_directory("90pcta")

        # # self.custom_strategy = load_strategy('BatsContest', self.config)
        # model_map = {
        #     "BTC/USDT": [
        #         ("20220405032644", "None.h5"),
        #     ],
        #     # 'ETH/USDT': [('20220403113648', 'none.h5')],
        # }
        # try:
        #     profile = Profile.load("BTC/USDT", "20220405075629")
        #     self.startup_candle_count = (
        #         profile.step_size * timeframe_to_minutes(profile.max_tf) // 60
        #     )
        #     for model in profile.available_models():
        #         self.models["BTC/USDT"].append((profile, model))
        # except Exception as e:
        #     logger.exception(e)
        #     raise
        # # for pair, model_info in model_map.items():
        # #     try:
        # #         self.models[pair].extend(self.load_model(pair, model_info))
        # #     except Exception as e:
        # #         logger.exception(f'Could not load model: {e}')
        # #     else:
        # #         logger.info(f'Loaded model: {model_info}')
        # self.validate_profiles()
        # if timeframe_to_minutes(self.timeframe) // 60 == 0:
        #     raise ValueError(
        #         f"Timeframes {self.timeframe} less than an hour are not supported."
        #         f" Please use a larger timeframe."
        #     )

    @staticmethod
    def load_model(
        pair: str, models_info: list[tuple[str, str]]
    ) -> list[tuple[Profile, Sequential]]:
        """
        Loads the models from the models folder and returns a list of dictionaries containing the
        profile and model

        :param pair: The currency pair to be traded
        :type pair: str
        :param models_info: A list of tuples, where each tuple contains a time and a model name
        :return: A list of dictionaries. Each dictionary contains a profile and a model.
        """
        # model = create_cnn(224)
        model_list = []
        pair = pair.replace("/", "_")
        for time, model_name in models_info:
            load = Profile.load(pair, time)
            model_list.append((load, load_model(REPO / pair / time / "models" / model_name)))

        return model_list

    def validate_profiles(self):
        """
        Validates the loaded profiles.
        """
        # timeframes = []
        for pair, models in self.models.items():
            for profile, model in models:
                try:
                    assert profile.pair == pair, "Profile pair does not match"
                    assert timeframe_to_minutes(self.timeframe) <= timeframe_to_minutes(
                        profile.min_tf
                    ), (
                        f"Smallest profile timeframe is {profile.min_tf} smaller than "
                        f"strategy timeframe {self.timeframe}"
                    )
                    # step_size in hours
                    if not self.padding:
                        self.padding = (
                            profile.step_size * timeframe_to_minutes(profile.max_tf) // 60
                        )
                        self.step_size = timeframe_to_minutes(profile.max_tf) // 60

                    # timeframes.append(profile.timeframes)
                except AssertionError as e:
                    raise ValueError(
                        f"Profile @ {profile.model_save_path} is not valid: {e}"
                    ) from e
        # make sure all profiles have the same timeframes
        # assert all(timeframes[0] == tf for tf in timeframes), 'All timeframes do not match'

    # @informative("1h")
    # def informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     return dataframe
    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        try:
            pairs = self.dp.current_whitelist()
        except OperationalException:
            return []
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        # Optionally Add additional "static" pairs
        informative_pairs = []
        for tf in ['15m', '30m', '1h']:
            informative_pairs.extend([(pair, tf) for pair in pairs])

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # logger.info(f'Calculating TA indicators for {metadata["pair"]}')
        #
        # # indicators = dataframe[dataframe.columns[~dataframe.columns.isin(COLUMNS_FILTER)]]
        # # assert all(indicators.max() < 1.00001) and all(
        # #     indicators.min() > -0.00001
        # # ), "Error, values are not normalized!"
        # # logger.info(f'{metadata["pair"]} - indicators populated!')
        # # dataframe = self.custom_strategy.populate_indicators(dataframe, metadata)
        # # rsi
        # dataframe[f'rsi'] = stats.zscore(ta.RSI(dataframe['close']))
        # # awesome oscillator
        # dataframe['ao'] = stats.zscore(
        #     pta.ao(dataframe['high'], dataframe['low'], fast=12, slow=26)
        # )
        #
        # # macd
        # macd, macdsignal, macdhist = ta.MACD(dataframe['close'])
        # dataframe['macd'] = stats.zscore(macd)
        # dataframe['macdsignal'] = stats.zscore(macdsignal)
        # dataframe['macdhist'] = stats.zscore(macdhist)
        #
        # # aroon
        # dataframe['aroonup'], dataframe['aroondown'] = stats.zscore(
        #     ta.AROON(dataframe['high'], dataframe['low'], timeperiod=25)
        # )
        # dataframe['current_price'] = stats.zscore(dataframe['close'])
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # dataframe['buy'] = self.rl_model_predict(dataframe)
        # assert self.models.get(metadata["pair"]) is not None, "Model is not loaded."
        logger.info(f'Populating buy signal for {metadata["pair"]}')
        # dataframe = self.predict(dataframe, metadata["pair"])
        dataframe = self.predict2(dataframe, metadata["pair"])

        print(dataframe["buy"].value_counts(), "buy signals")
        logger.info(f'{metadata["pair"]} - buy signal populated!')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # logger.info(f'Populating sell signal for {metadata["pair"]}')
        # # action = self.rl_model_predict(dataframe, metadata['pair'])
        # # dataframe['sell'] = (action == 2).astype('int')
        # # print number of sell signals
        # print(dataframe["sell"].value_counts(), "sell signals")
        # logger.info(f'{metadata["pair"]} - sell signal populated!')
        return dataframe

    def should_sell(
        self,
        trade: Trade,
        rate: float,
        current_time: datetime,
        buy: bool,
        sell: bool,
        low: float = None,
        high: float = None,
        force_stoploss: float = 0,
    ) -> SellCheckTuple:
        # if current_time - trade.open_date >= timedelta(days=1):
        #     return SellCheckTuple(SellType.CUSTOM_SELL, "1 day passed")
        if current_time >= trade.open_date + timedelta(minutes=timeframe_to_minutes(self.max_tf)):
            return SellCheckTuple(SellType.SELL_SIGNAL, "")
        return super().should_sell(trade, rate, current_time, buy, sell, low, high, force_stoploss)

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle["close"]

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle["close"]

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        sell_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        return super().confirm_trade_exit(
            pair,
            trade,
            order_type,
            amount,
            rate,
            time_in_force,
            sell_reason,
            current_time,
            **kwargs,
        )

    # def custom_sell(
    #     self,
    #     pair: str,
    #     trade: Trade,
    #     current_time: datetime,
    #     current_rate: float,
    #     current_profit: float,
    #     **kwargs,
    # ) -> Optional[Union[str, bool]]:
    #     # trade_date = trade.open_date.date()
    #     if current_time - trade.open_date >= timedelta(days=1):
    #         return True

    def calculate_candles_needed(self, days: int):
        """
        Calculate the number of candles needed to cover the given number of days

        :param days: The number of days to look back
        :return: The number of candles needed to cover the number of days specified.
        """
        MINUTES_IN_DAY = 1440
        tf_minutes = timeframe_to_minutes(self.timeframe)
        return days * MINUTES_IN_DAY // tf_minutes

    # def preprocess(self, indicators: pd.DataFrame, profile: Profile):
    #     # t1 = time.perf_counter()
    #     folder = preprocess(profile, Mode.PREDICT, indicators)
    #     gen = get_single_generator(profile, folder=folder)
    #     model_input = gen.next()[0]
    #     # images = quick_gaf(indicators)[0]
    #     # images = tensor_transform(images[-1])
    #     # print('preprocess() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
    #     return model_input, folder

    @staticmethod
    def preprocess(indicators: pd.DataFrame, profile: Profile, custom_timeframes: dict = None):
        # folder, dates = preprocess(profile, Mode.PREDICT, indicators, custom_timeframes)
        # model_input = get_single_generator(profile, folder=folder, dates=dates)
        # return model_input, dates, folder
        data_input = preprocess(profile, Mode.PREDICT, indicators, custom_timeframes)
        return data_input

    # def rl_model_predict2(self, dataframe: DataFrame, pair: str):
    #     df = dataframe.iloc[self.startup_candle_count :]
    #     logging.info(f'Predicting buy/sell for {pair}')
    #     action_output = pd.DataFrame(np.zeros((len(dataframe), 2)))
    #     # indicators = dataframe.copy()[['date', 'open', 'close']]
    #     step_size = ...
    #
    #     gb = df.groupby(np.arange(len(df)) // self.step_size)
    #
    #     for index, data in gb:
    #         print(index, data)
    #         buy_results = []
    #         sell_results = []
    #         for profile, model in self.models[pair]:
    #             observation = self.preprocess(data, profile)
    #             # t1 = time.perf_counter()
    #             buy_results.append(model.predict(observation)[0][0])
    #             sell_results.append(model.predict(observation)[0][1])
    #         buy_res = np.mean(buy_results)
    #         sell_res = np.mean(sell_results)
    #         # print('model.predict() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
    #         dataframe.iloc[data.iloc[-1].index]['buy', 'sell'] = [
    #             (1 if buy_res > 60 else 0),
    #             (1 if sell_res > 50 else 0),
    #         ]
    #     return dataframe

    # def rl_model_predict(self, dataframe: DataFrame, pair: str):
    #     logging.info(f'Predicting buy/sell for {pair}')
    #     action_output = pd.DataFrame(np.zeros((len(dataframe), 2)))
    #     # multiplier_output = pd.DataFrame(np.zeros((len(dataframe), 1)))
    #
    #     # indicators =
    #     # for c in COLUMNS_FILTER:
    #     #     # remove every column that contains a substring of c
    #     #     indicators = indicators.drop(columns=[col for col in indicators.columns if c in col])
    #     # indicators = dataframe.copy()[['date', 'open', 'close']]
    #     # start index where all indicators are available
    #     # print(f'{indicators.shape}')
    #     #  TODO: This is slow and ugly, must use .rolling
    #     df = dataframe.iloc[self.startup_candle_count - self.padding :]
    #     with alive_bar(
    #         (len(dataframe) - self.startup_candle_count) // self.step_size
    #     ) as bar:
    #         for window in range(self.startup_candle_count, len(dataframe), self.step_size):
    #             buy_results = []
    #             sell_results = []
    #             start = min(window - self.padding, len(dataframe) - self.padding)
    #             end = window
    #             for profile, model_name in self.models[pair]:
    #                 clear_session()
    #                 model = profile.get_model(model_name)
    #                 data_slice = dataframe[start:end]
    #                 try:
    #                     # observation, folder = self.preprocess(data_slice, profile)
    #                     observation = self.preprocess2(data_slice, profile)
    #                 except ValueError:
    #                     logger.exception(
    #                         f"There wasn't enough data to preprocess for {pair} at {start}"
    #                     )
    #                     break
    #                 # t1 = time.perf_counter()
    #                 prediction = model.predict(observation)[0]
    #                 # shutil.rmtree(folder)
    #                 buy_results.append(prediction[0])
    #                 sell_results.append(prediction[1])
    #             buy_res = statistics.mean(buy_results)
    #             sell_res = statistics.mean(sell_results)
    #             # print('model.predict() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
    #             action_output.loc[end] = [buy_res, sell_res]
    #             if len(self.models[pair]) > 1:
    #                 dataframe.loc[end, 'buy'] = 1 if buy_res > 0.5 else 0
    #
    #             else:
    #                 dataframe.loc[end, 'buy'] = round(buy_res)
    #             bar()
    #     return dataframe

    # def rl_model_predict2(self, dataframe: DataFrame, pair: str):
    #     logging.info(f"Predicting buy/sell for {pair}")
    #
    #     with alive_bar(len(self.models[pair]), title="Predicting....") as bar:
    #         results: dict[int, list] = defaultdict(list)
    #         for profile, model_name in self.models[pair]:
    #             clear_session()
    #             model = profile.get_model(model_name)
    #             for window in range(
    #                 self.startup_candle_count, len(dataframe), self.step_size
    #             ):
    #                 start = min(window - self.padding, len(dataframe) - self.padding)
    #                 end = window
    #                 data_slice = dataframe[start:end]
    #                 try:
    #                     # observation, folder = self.preprocess(data_slice, profile)
    #                     observation = self.preprocess2(data_slice, profile)
    #                 except ValueError:
    #                     logger.exception(
    #                         f"There wasn't enough data to preprocess for {pair} at {start}"
    #                     )
    #                     break
    #                 # t1 = time.perf_counter()
    #                 prediction = model.predict(observation)[0]
    #                 # shutil.rmtree(folder)
    #                 results[end].append(prediction[0])
    #             bar()
    #         for end, buy_results in results.items():
    #             buy_res = statistics.mean(buy_results)
    #             dataframe.loc[end, "buy"] = (
    #                 1 if round(buy_res) > self.min_buy_threshold else 0
    #             )
    #     return dataframe

    # def rl_model_predict3(self, dataframe: DataFrame, pair: str):
    #     logging.info(f"Predicting buy/sell for {pair}")
    #     n_models = 1
    #     with alive_bar(n_models, title="Predicting....") as bar:
    #         results: dict[int, list] = defaultdict(list)
    #         models = load_specific_tuner_model("tuner/20220407034952")
    #         # models = load_tuner_models()
    #         for profile, model in models:
    #             for window in range(self.startup_candle_count, len(dataframe), self.step_size):
    #                 start = min(window - self.padding, len(dataframe) - self.padding)
    #                 end = window
    #                 data_slice = dataframe[start : end + 1]
    #                 try:
    #                     # observation, folder = self.preprocess(data_slice, profile)
    #                     observation, folder = self.preprocess(data_slice, profile)
    #                 except ValueError:
    #                     logger.exception(
    #                         f"There wasn't enough data to preprocess for {pair} at {start}"
    #                     )
    #                     break
    #                 # t1 = time.perf_counter()
    #                 prediction = model.predict(observation)[0]
    #                 shutil.rmtree(folder)
    #                 results[data_slice.iloc[-1]["date"]].append(round(prediction[0]))
    #             clear_session()
    #         bar()
    #     for date, buy_results in results.items():
    #         buy_res = statistics.mean(buy_results)
    #         try:
    #             dataframe.loc[dataframe["date"].dt.date == date, "buy"] = (
    #                 1 if round(buy_res) > 0.60 else 0
    #             )
    #         except (KeyError, IndexError):
    #             logger.info(f"No data for {date}")
    #             break
    #     return dataframe
    @staticmethod
    def preprocess2(indicators: list[pd.DataFrame], profile: Profile):
        # folder, dates = preprocess(profile, Mode.PREDICT, indicators, custom_timeframes)
        # model_input = get_single_generator(profile, folder=folder, dates=dates)
        # return model_input, dates, folder
        profile.ensure_directories_exist()
        test_dates = preprocess_train2(indicators, Mode.PREDICT, profile)
        gen = get_single_generator(
            profile, folder=profile.get_images_path(Mode.PREDICT), dates=test_dates
        )
        return gen, test_dates

    # def rl_model_predict2(self, dataframe: DataFrame, pair: str):
    #     df = dataframe.iloc[self.startup_candle_count :]
    #     logging.info(f'Predicting buy/sell for {pair}')
    #     action_output = pd.DataFrame(np.zeros((len(dataframe), 2)))
    #     # indicators = dataframe.copy()[['date', 'open', 'close']]
    #     step_size = ...
    #
    #     gb = df.groupby(np.arange(len(df)) // self.step_size)
    #
    #     for index, data in gb:
    #         print(index, data)
    #         buy_results = []
    #         sell_results = []
    #         for profile, model in self.models[pair]:
    #             observation = self.preprocess(data, profile)
    #             # t1 = time.perf_counter()
    #             buy_results.append(model.predict(observation)[0][0])
    #             sell_results.append(model.predict(observation)[0][1])
    #         buy_res = np.mean(buy_results)
    #         sell_res = np.mean(sell_results)
    #         # print('model.predict() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
    #         dataframe.iloc[data.iloc[-1].index]['buy', 'sell'] = [
    #             (1 if buy_res > 60 else 0),
    #             (1 if sell_res > 50 else 0),
    #         ]
    #     return dataframe

    # def rl_model_predict(self, dataframe: DataFrame, pair: str):
    #     logging.info(f'Predicting buy/sell for {pair}')
    #     action_output = pd.DataFrame(np.zeros((len(dataframe), 2)))
    #     # multiplier_output = pd.DataFrame(np.zeros((len(dataframe), 1)))
    #
    #     # indicators =
    #     # for c in COLUMNS_FILTER:
    #     #     # remove every column that contains a substring of c
    #     #     indicators = indicators.drop(columns=[col for col in indicators.columns if c in col])
    #     # indicators = dataframe.copy()[['date', 'open', 'close']]
    #     # start index where all indicators are available
    #     # print(f'{indicators.shape}')
    #     #  TODO: This is slow and ugly, must use .rolling
    #     df = dataframe.iloc[self.startup_candle_count - self.padding :]
    #     with alive_bar(
    #         (len(dataframe) - self.startup_candle_count) // self.step_size
    #     ) as bar:
    #         for window in range(self.startup_candle_count, len(dataframe), self.step_size):
    #             buy_results = []
    #             sell_results = []
    #             start = min(window - self.padding, len(dataframe) - self.padding)
    #             end = window
    #             for profile, model_name in self.models[pair]:
    #                 clear_session()
    #                 model = profile.get_model(model_name)
    #                 data_slice = dataframe[start:end]
    #                 try:
    #                     # observation, folder = self.preprocess(data_slice, profile)
    #                     observation = self.preprocess2(data_slice, profile)
    #                 except ValueError:
    #                     logger.exception(
    #                         f"There wasn't enough data to preprocess for {pair} at {start}"
    #                     )
    #                     break
    #                 # t1 = time.perf_counter()
    #                 prediction = model.predict(observation)[0]
    #                 # shutil.rmtree(folder)
    #                 buy_results.append(prediction[0])
    #                 sell_results.append(prediction[1])
    #             buy_res = statistics.mean(buy_results)
    #             sell_res = statistics.mean(sell_results)
    #             # print('model.predict() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
    #             action_output.loc[end] = [buy_res, sell_res]
    #             if len(self.models[pair]) > 1:
    #                 dataframe.loc[end, 'buy'] = 1 if buy_res > 0.5 else 0
    #
    #             else:
    #                 dataframe.loc[end, 'buy'] = round(buy_res)
    #             bar()
    #     return dataframe

    # def rl_model_predict2(self, dataframe: DataFrame, pair: str):
    #     logging.info(f"Predicting buy/sell for {pair}")
    #
    #     with alive_bar(len(self.models[pair]), title="Predicting....") as bar:
    #         results: dict[int, list] = defaultdict(list)
    #         for profile, model_name in self.models[pair]:
    #             clear_session()
    #             model = profile.get_model(model_name)
    #             for window in range(
    #                 self.startup_candle_count, len(dataframe), self.step_size
    #             ):
    #                 start = min(window - self.padding, len(dataframe) - self.padding)
    #                 end = window
    #                 data_slice = dataframe[start:end]
    #                 try:
    #                     # observation, folder = self.preprocess(data_slice, profile)
    #                     observation = self.preprocess2(data_slice, profile)
    #                 except ValueError:
    #                     logger.exception(
    #                         f"There wasn't enough data to preprocess for {pair} at {start}"
    #                     )
    #                     break
    #                 # t1 = time.perf_counter()
    #                 prediction = model.predict(observation)[0]
    #                 # shutil.rmtree(folder)
    #                 results[end].append(prediction[0])
    #             bar()
    #         for end, buy_results in results.items():
    #             buy_res = statistics.mean(buy_results)
    #             dataframe.loc[end, "buy"] = (
    #                 1 if round(buy_res) > self.min_buy_threshold else 0
    #             )
    #     return dataframe

    # def rl_model_predict3(self, dataframe: DataFrame, pair: str):
    #     logging.info(f"Predicting buy/sell for {pair}")
    #     n_models = 1
    #     with alive_bar(n_models, title="Predicting....") as bar:
    #         results: dict[int, list] = defaultdict(list)
    #         models = load_specific_tuner_model("tuner/20220407034952")
    #         # models = load_tuner_models()
    #         for profile, model in models:
    #             for window in range(self.startup_candle_count, len(dataframe), self.step_size):
    #                 start = min(window - self.padding, len(dataframe) - self.padding)
    #                 end = window
    #                 data_slice = dataframe[start : end + 1]
    #                 try:
    #                     # observation, folder = self.preprocess(data_slice, profile)
    #                     observation, folder = self.preprocess(data_slice, profile)
    #                 except ValueError:
    #                     logger.exception(
    #                         f"There wasn't enough data to preprocess for {pair} at {start}"
    #                     )
    #                     break
    #                 # t1 = time.perf_counter()
    #                 prediction = model.predict(observation)[0]
    #                 shutil.rmtree(folder)
    #                 results[data_slice.iloc[-1]["date"]].append(round(prediction[0]))
    #             clear_session()
    #         bar()
    #     for date, buy_results in results.items():
    #         buy_res = statistics.mean(buy_results)
    #         try:
    #             dataframe.loc[dataframe["date"].dt.date == date, "buy"] = (
    #                 1 if round(buy_res) > 0.60 else 0
    #             )
    #         except (KeyError, IndexError):
    #             logger.info(f"No data for {date}")
    #             break
    #     return dataframe

    def predict(self, dataframe: DataFrame, pair: str):
        """
        It takes a dataframe, and a pair, and then it iterates through the models, and for each model,
        it preprocesses the dataframe, and then it predicts the buy/sell signal for each date in the
        dataframe

        :param dataframe: The dataframe containing the data for the pair you want to predict
        :type dataframe: DataFrame
        :param pair: The pair to be traded
        :type pair: str
        :return: The model is being returned.
        """
        logging.info(f"Predicting buy/sell for {pair}")
        n_models = 1
        results: dict[int, list] = defaultdict(list)

        # models = load_tuner_models()
        model_inputs = None
        dates = None
        for profile, model in self.models_:
            logger.info(f"Predicting using {profile}")
            assert self.timeframe == profile.min_tf
            self.max_tf = profile.max_tf
            custom_timeframes = self.create_custom_timeframes(pair, profile)
            custom_timeframes.insert(0, dataframe)
            if not model_inputs:
                # model_inputs, dates, _ = self.preprocess(dataframe, profile)
                model_inputs = self.preprocess(dataframe, profile)
            # answers = model((model_inputs.values()), training=False)
            for date, inp in model_inputs.items():
                answer = model(inp, training=False)
                results[date].append(float(answer[0, 0]))
            # for date, answer in zip(model_inputs.keys(), [a for a in answers.numpy()]):
            #     results[date].append(answer[0])
            clear_session()
            model_inputs = None
        # if folder:
        #     shutil.rmtree(folder)
        #     print(f"Deleted -> {folder}")

        for date, buy_results in results.items():
            buy_res = statistics.mean([round(r) for r in buy_results])
            try:
                res = 1 if round(buy_res) > self.min_buy_threshold else 0
                print(f"{date} -> {buy_results} -> {buy_res} -> {round(res)}")
                dataframe.loc[dataframe["date"] == date, "buy"] = res
            except (KeyError, IndexError):
                logger.info(f"No data for {date}")
                break
        dataframe["buy"] = dataframe["buy"].shift(-1)
        return dataframe

    def predict2(self, dataframe: DataFrame, pair: str):
        """
        It takes a dataframe, and a pair, and then it iterates through the models, and for each model,
        it preprocesses the dataframe, and then it predicts the buy/sell signal for each date in the
        dataframe

        :param dataframe: The dataframe containing the data for the pair you want to predict
        :type dataframe: DataFrame
        :param pair: The pair to be traded
        :type pair: str
        :return: The model is being returned.
        """
        logging.info(f"Predicting buy/sell for {pair}")
        n_models = 1
        results: dict[int, list] = defaultdict(list)

        # models = load_tuner_models()
        model_inputs = None
        dates = None
        for profile, model in self.models_:
            logger.info(f"Predicting using {profile}")
            assert self.timeframe == profile.min_tf
            self.max_tf = profile.max_tf
            custom_timeframes = self.create_custom_timeframes(pair, profile)
            custom_timeframes.insert(0, dataframe[['date', 'close', 'open']])
            # if not model_inputs:
            # model_inputs, dates, _ = self.preprocess(dataframe, profile)
            gen, dates = self.preprocess2(custom_timeframes, profile)
            # answers = model((model_inputs.values()), training=False)
            answers = model.predict(gen)
            for date, answer in zip(dates, answers):
                buy_answer = answer[0]
                results[date].append(float(buy_answer))
            # for date, answer in zip(model_inputs.keys(), [a for a in answers.numpy()]):
            #     results[date].append(answer[0])
            clear_session()
            model_inputs = None
        # if folder:
        #     shutil.rmtree(folder)
        #     print(f"Deleted -> {folder}")

        for date, buy_results in results.items():
            buy_res = statistics.mean([round(r) for r in buy_results])
            try:
                res = 1 if round(buy_res) > self.min_buy_threshold else 0
                print(f"{date} -> {buy_results} -> {buy_res} -> {round(res)}")
                dataframe.loc[dataframe["date"] == date, "buy"] = res
            except (KeyError, IndexError):
                logger.info(f"No data for {date}")
                break
        # dataframe["buy"] = dataframe["buy"].shift(-1)
        return dataframe

    def create_custom_timeframes(self, pair, profile):
        custom_timeframes = []
        initial_startup_candle_count = self.config["startup_candle_count"]
        for tf in profile.timeframes[1:]:
            self.config["startup_candle_count"] = (
                profile.window_size / (timeframe_to_minutes(tf) / 60) * 24
            )
            custom_timeframes.append(
                self.dp.get_pair_dataframe(pair=pair, timeframe=tf)[["date", "close", "open"]]
            )
        self.config["startup_candle_count"] = initial_startup_candle_count
        return custom_timeframes

    # def custom_stake_amount(
    #     self,
    #     pair: str,
    #     current_time: datetime,
    #     current_rate: float,
    #     proposed_stake: float,
    #     min_stake: float,
    #     max_stake: float,
    #     entry_tag: Optional[str],
    #     **kwargs,
    # ) -> float:
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = dataframe.iloc[-1].squeeze()
    #     pob = self.percent_of_balance_dict[last_candle.name.astype('int')]
    #     if pob > 0:
    #         return pob / 10 * self.wallets.get_available_stake_amount()


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    return timedelta(minutes=timeframe_to_minutes(timeframe))


def timeframe_to_hours(timeframe: str) -> int:
    """
    Return the number of hours in the given timeframe

    :param timeframe: The timeframe to use
    :return: The number of hours in the timeframe.
    """
    return timeframe_to_minutes(timeframe) // 60


# load_tuner_models()
