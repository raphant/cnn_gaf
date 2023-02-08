# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import logging
from pathlib import Path

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from stable_baselines3.ppo.ppo import PPO

import predict

logger = logging.getLogger(__name__)

COLUMNS_FILTER = [
    'date',
    'open',
    'close',
    'high',
    'low',
    'volume',
    'buy',
    'sell',
    'buy_tag',
    'exit_tag',
]


class SagesFreqGym(IStrategy):

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

    ticker_interval = '5m'

    use_sell_signal = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    startup_candle_count: int = 200

    model = None
    window_size = None

    timeperiods = [7, 14, 21]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        try:
            # get the file that starts with "best_model_" in the models/ directory
            # list files in the directory
            # files = Path('models/').glob('final_model_*')
            # get the first file
            # model_file = next(files)
            model_file = Path(
                'models/best_model_FreqGymNormalized_FreqtradeEnv_PPO_20220317_064037.zip'
            )
            assert model_file.exists(), f'Model file "{model_file}" does not exist.'
            self.model = PPO.load(
                str(model_file)
            )  # Note: Make sure you use the same policy as the one used to train
            self.window_size = self.model.observation_space.shape[0]
        except Exception as e:
            logger.exception(f'Could not load model: {e}')
            raise
        else:
            logger.info(f'Loaded model: {model_file}')

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
        logger.info(f'Calculating TA indicators for {metadata["pair"]}')
        for period in [10, 20, 40, 80, 160]:
            # wma
            dataframe[f'wma_{period}'] = ta.WMA(dataframe, timeperiod=period)
            # rsi
            dataframe[f'rsi_{period}'] = ta.RSI(dataframe, timeperiod=period)
            # rvi
            dataframe[f'rvi_{period}'] = ta.RVI(dataframe, timeperiod=period)

        # volume roc
        dataframe['volume_roc'] = ta.ROC(dataframe['volume'], timeperiod=1)
        # indicators = dataframe[dataframe.columns[~dataframe.columns.isin(COLUMNS_FILTER)]]
        # assert all(indicators.max() < 1.00001) and all(
        #     indicators.min() > -0.00001
        # ), "Error, values are not normalized!"
        # logger.info(f'{metadata["pair"]} - indicators populated!')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # dataframe['buy'] = self.rl_model_predict(dataframe)
        logger.info(f'Populating buy signal for {metadata["pair"]}')
        action = self.rl_model_predict(dataframe)
        dataframe['buy'] = (action == 1).astype('int')

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
        action = self.rl_model_predict(dataframe)
        dataframe['sell'] = (action == 2).astype('int')
        logger.info(f'{metadata["pair"]} - sell signal populated!')
        return dataframe

    def rl_model_predict(self, dataframe):
        output = pd.DataFrame(np.zeros((len(dataframe), 1)))

        indicators = (
            dataframe[dataframe.columns[~dataframe.columns.isin(COLUMNS_FILTER)]]
            .fillna(0)
            .to_numpy()
        )

        #  TODO: This is slow and ugly, must use .rolling
        for window in range(self.window_size, len(dataframe)):
            start = window - self.window_size
            end = window
            observation = indicators[start:end]
            res, _ = predict.predict(observation, deterministic=True)
            output.loc[end] = res

        return output


def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)
