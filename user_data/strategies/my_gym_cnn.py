# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import logging
import time
from datetime import timedelta
from functools import partial

import numpy as np  # noqa
import pandas as pd  # noqa
from freqtrade.strategy.interface import IStrategy
from keras import Sequential
from keras.models import load_model
from pandas import DataFrame

from time_series_to_gaf.cnn_model import create_cnn
from time_series_to_gaf.constants import REPO
from time_series_to_gaf.preprocess import quick_gaf, tensor_transform

logger = logging.getLogger(__name__)

COLUMNS_FILTER = [
    'date',
    'open',
    'close',
    'high',
    'low',
    'buy',
    'sell',
    'volume',
    'buy_tag',
    'exit_tag',
]

from scipy import stats

stats.zscore = partial(stats.zscore, nan_policy='omit')


class SagesGymCNN(IStrategy):
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

    timeframe = '1h'

    use_sell_signal = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    startup_candle_count: int = 504

    models: dict[str, Sequential] = {}
    window_size = 504

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # self.custom_strategy = load_strategy('BatsContest', self.config)
        model_map = {
            'BTC/USDT': ('20220403125044', 'RandomUniform.h5'),
            'ETH/USDT': ('20220403113648', 'none.h5'),
            'LTC/USDT': ('20220403120134', 'LecunUniform.h5'),
        }
        for pair, model_info in model_map.items():
            try:
                self.models[pair] = self.load_model(pair, *model_info)
            except Exception as e:
                raise RuntimeError(f'Could not load model: {e}') from e
            else:
                logger.info(f'Loaded model: {model_info}')

    # @informative('1h')
    # def informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     return dataframe
    def load_model(self, pair: str, time: str, model_name: str) -> Sequential:
        # model = create_cnn(224)
        model_to_load = REPO / pair.replace('/', '_') / time / 'models' / model_name

        return load_model(model_to_load)

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
        assert self.models.get(metadata['pair']) is not None, 'Model is not loaded.'
        logger.info(f'Populating buy signal for {metadata["pair"]}')
        action = self.rl_model_predict(dataframe, metadata['pair'])
        dataframe['buy'] = (action[0] > 0.50).astype('int')
        dataframe['sell'] = (action[1] > 0.50).astype('int')

        print(dataframe['buy'].value_counts(), 'buy signals')
        logger.info(f'{metadata["pair"]} - buy signal populated!')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        logger.info(f'Populating sell signal for {metadata["pair"]}')
        # action = self.rl_model_predict(dataframe, metadata['pair'])
        # dataframe['sell'] = (action == 2).astype('int')
        # print number of sell signals
        print(dataframe['sell'].value_counts(), 'sell signals')
        logger.info(f'{metadata["pair"]} - sell signal populated!')
        return dataframe

    def preprocess(self, indicators: pd.DataFrame):
        t1 = time.perf_counter()
        images = quick_gaf(indicators)[0]
        images = tensor_transform(images[-1])
        print('preprocess() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
        return images

    def rl_model_predict(self, dataframe: DataFrame, pair: str):
        action_output = pd.DataFrame(np.zeros((len(dataframe), 2)))
        # multiplier_output = pd.DataFrame(np.zeros((len(dataframe), 1)))

        # indicators =
        # for c in COLUMNS_FILTER:
        #     # remove every column that contains a substring of c
        #     indicators = indicators.drop(columns=[col for col in indicators.columns if c in col])
        indicators = dataframe.copy()[['date', 'open', 'close']]
        # start index where all indicators are available
        # print(f'{indicators.shape}')
        #  TODO: This is slow and ugly, must use .rolling
        for window in range(self.window_size, len(dataframe), 24):
            start = window - self.window_size
            end = window
            observation = self.preprocess(indicators[start:end])
            t1 = time.perf_counter()
            res = self.models[pair].predict(observation)[0]
            print('model.predict() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
            action_output.loc[end] = res

        return action_output

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
