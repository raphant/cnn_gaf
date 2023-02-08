import datetime
from collections import defaultdict
from functools import partial
from typing import Any, Iterable

import pandas as pd
from alive_progress import alive_bar
from diskcache import Cache
from lazyft import paths

import util

cache = Cache(paths.CACHE_DIR / "split")
alive_bar = partial(alive_bar, force_tty=True)


@util.timeit
def create_decision_map(
    df,
    list_dates: list[Any],
    timeframes: list[str],
    window: int,
    download_interval: str,
):
    """
    Given a dataframe, a list of dates, a list of timeframes, and a window,
    the function will create a dictionary of decisions mapped to a list of
    gafs

    :param df: The dataframe containing the data
    :param list_dates: a list of dates that we want to create a map for
    :param timeframes: a list of timeframes to group the data by
    :param window: The number of days to look back in the past to make the decision
    :param download_interval: The interval used to download the data
    :return: A dictionary of decisions mapped to a list of gafs.
    """
    decision_map = defaultdict(list)
    index = window
    with alive_bar(
        len(list_dates) - window - 1,
        title="Creating decision map...",
        bar="filling",
    ) as bar:
        while index <= len(list_dates):
            data_slice, save_idx = create_data_slice(index, df, window, list_dates)
            tf_series = create_tf_series_list(data_slice, download_interval, timeframes)
            try:
                decision = get_trading_decision(df, list_dates, save_idx)
            except IndexError:
                break
            decision_map[decision].append(
                [list_dates[save_idx].strftime("%Y%m%d%H%M"), tf_series]
            )
            print(
                f"[{data_slice.iloc[0]['date']} to {data_slice.iloc[-1]['date']}] - "
                f"{pd.to_datetime(list_dates[index])} Prediction -> {decision} | "
                f"Hash: {util.hash(tf_series)}"
            )
            index += 1
            bar()
            # print()
    return decision_map


@util.timeit
def create_decision_map_with_custom_timeframes(
    dfs: dict[str, pd.DataFrame],
    max_tf: str,
    list_dates: list[Any],
    window: int,
    tail_window: int = 20,
):
    """
    Given a dataframe, a list of dates, a list of timeframes, and a window,
    the function will create a dictionary of decisions mapped to a list of
    gafs

    :param dfs: The dataframes containing the data
    :param list_dates: a list of dates that we want to create a map for
    :param window: The number of days to look back in the past to make the decision
    :return: A dictionary of decisions mapped to a list of gafs.
    """
    decision_map = defaultdict(list)
    index = window
    with alive_bar(
        len(list_dates) - window - 1,
        title="Creating decision map...",
        bar="filling",
    ) as bar:
        while index <= len(list_dates):
            data_slices, save_idx = create_data_slices(
                index, dfs.values(), window, list_dates
            )
            tf_series = create_tf_series_list2(data_slices, window)
            try:
                decision = get_trading_decision(dfs[max_tf], list_dates, save_idx)
            except IndexError:
                break
            if any(d.empty for d in data_slices):
                print(f"Found empty dataframes for {list_dates[index]}--... Skipping")
                index += 1
                bar()
                continue
            decision_map[decision].append(
                [list_dates[save_idx].strftime("%Y%m%d%H%M"), tf_series]
            )
            print(
                f"[{data_slices[-1].iloc[0]['date']} to {data_slices[-1].iloc[-1]['date']}] - "
                f"{pd.to_datetime(list_dates[index])} Prediction -> {decision} | "
                f"Hash: {util.hash(tf_series)}"
            )
            index += 1
            bar()
            # print()
    return decision_map


# @cache.memoize()
def split_timeframes(
    df: pd.DataFrame,
    window: int,
    timeframes: list[str],
    list_dates: list[Any],
    download_interval: str,
    custom_timeframes: dict[str, pd.DataFrame] = None,
) -> dict[str, list[pd.Series]]:
    """
    Given a dataframe, a window size, a list of timeframes, and a list of dates,
    split the dataframe into a list of lists of series, where each series is a list of prices
    for a specific timeframe

    :param df: The dataframe that contains the data
    :param window: The size of the moving window. This is the number of observations used for
    calculating the statistic. Each window will be a fixed size
    :param timeframes: a list of timeframes to use for grouping the data
    :param list_dates: list of dates
    :param download_interval: the interval that was used to download the data
    :param custom_timeframes: a dictionary of custom timeframes to use for grouping the data
    :return: A list of lists of series. Each list of series is a list of the closing prices of the
    stocks for a given timeframe.
    """
    series_dict: dict[str, list[pd.Series]] = {}
    index = window
    with alive_bar(
        len(list_dates) - window + 1, title="Creating decision map...", bar="filling"
    ) as bar:
        while index <= len(list_dates):
            data_slice, save_idx = create_data_slice(index, df, window, list_dates)
            tf_series = create_tf_series_list(data_slice, download_interval, timeframes)
            print(
                f"[{pd.to_datetime(list_dates[save_idx])}] {timeframes[-1]} - "
                f'{data_slice.iloc[0]["date"]} to {data_slice.iloc[-1]["date"]} | '
                f"Hash: {util.hash(tf_series)}"
            )
            series_dict[list_dates[save_idx].strftime("%Y%m%d%H%M")] = tf_series
            index += 1
            bar()
    return series_dict


# @cache.memoize()
def split_timeframes2(
    dfs: dict[str, pd.DataFrame],
    window: int,
    list_dates: list[Any],
    tail_window: int = 20,
) -> dict[str, list[pd.Series]]:
    """
    Given a dataframe, a window size, a list of timeframes, and a list of dates,
    split the dataframe into a list of lists of series, where each series is a list of prices
    for a specific timeframe

    :return: A list of lists of series. Each list of series is a list of the closing prices of the
    stocks for a given timeframe.
    """
    series_dict: dict[str, list[pd.Series]] = {}
    index = window
    with alive_bar(
        len(list_dates) - window + 1, title="Creating decision map...", bar="filling"
    ) as bar:
        while index <= len(list_dates):
            data_slices, save_idx = create_data_slices(
                index, dfs.values(), window, list_dates
            )
            tf_series = create_tf_series_list2(data_slices, tail_window)
            print(
                f"[{data_slices[-1].iloc[0]['date']} to {data_slices[-1].iloc[-1]['date']}] - "
                f"{pd.to_datetime(list_dates[index])}"
                f"Hash: {util.hash(tf_series)}"
            )
            series_dict[list_dates[save_idx].strftime("%Y%m%d%H%M")] = tf_series
            index += 1
            bar()
    return series_dict


def get_trading_decision(
    df: pd.DataFrame, list_dates: list[datetime.datetime], save_idx: int
):
    """
    Given a dataframe, a list of dates, and an index, return the trading decision for the next day

    :param df: the dataframe of the stock
    :param list_dates: list of dates that we want to predict
    :param save_idx: the index of the date in the list_dates list that we want to predict the trading
    decision for
    :return: The trading decision for the next day.
    """
    predict_slice = df.loc[
        (df["date"] >= list_dates[save_idx]) & (df["date"] < list_dates[save_idx + 1])
    ]
    future_open = predict_slice.iloc[0]["open"]
    future_close = predict_slice.iloc[-1]["close"]
    return trading_action_new(future_open, future_close)


def create_data_slice(
    index: int, df: pd.DataFrame, window: int, list_dates: list[datetime.datetime]
):
    """
    This function takes in an index, a dataframe, a window size, and a list of dates. It then creates
    a data slice of the dataframe that is between the index and the index minus the window size

    :param index: the index of the date we're looking at
    :param df: the dataframe containing the data
    :param window: the number of days to look back
    :param list_dates: a list of dates that we want to use as the end date of our data slices
    :return: A dataframe and an index
    """
    idx = index
    save_idx = index
    if save_idx >= len(list_dates) - 1:
        save_idx = -1
        after = df["date"] >= list_dates[-window - 1]
        before = df["date"] < list_dates[-1]
    else:
        after = df["date"] >= list_dates[idx - window]
        before = df["date"] < list_dates[idx]
    data_slice = df.loc[after & before]
    return data_slice, save_idx


def create_data_slices(
    index: int,
    dfs: Iterable[pd.DataFrame],
    window: int,
    list_dates: list[datetime.datetime],
):
    """
    This function takes in an index, a dataframe, a window size, and a list of dates. It then creates
    a data slice of the dataframe that is between the index and the index minus the window size

    :param index: An index used for locating the datestamp in list_dates
    :param dfs: A list of dataframes at various timeframes
    :param window: the number of candles to look back
    :param list_dates: a list of dates that we want to use as the end date of our data slices
    :return: A dataframe and the index key for the date used to save the data slices
    """
    slices = []
    save_idx = index
    idx = index
    for df in dfs:
        if idx >= len(list_dates) - 1:
            save_idx = -1
            after = df["date"] >= list_dates[-window - 1]
            before = df["date"] < list_dates[-1]
        else:
            after = df["date"] >= list_dates[idx - window]
            before = df["date"] < list_dates[idx]
        slices.append(df.loc[after & before])
    return slices, save_idx


def create_tf_series_list(
    data_slice: pd.DataFrame, download_interval: str, timeframes: list[str]
) -> list[pd.Series]:
    """
    Takes an ohlc within a certain timerange and returns a list of Series containing the last 20
    candles from each timeframe.

    :param data_slice: the dataframe slice that we're going to use to create the tf_series
    :param download_interval: the interval of the data you're downloading
    :param timeframes: list of timeframes to create
    :return: A list of series, each series is the close price for a given timeframe.
    """
    tf_series: list[pd.Series] = []
    for freq in timeframes:
        if freq == download_interval:
            append = data_slice.tail(20)
            tf_series.append(append["close"])
            assert len(tf_series[-1]) == 20, (
                f"tf_series for {freq} is not 20\n"
                f"tf_series len: {len(tf_series[-1])}\n"
                f"data_slice tail: {data_slice.tail()}\n"
            )
            # print(
            #     f"[{pd.to_datetime(list_dates[save_idx])}] {freq} - "
            #     f'{append.iloc[0]["date"]} to {append.iloc[-1]["date"]}'
            #     f" | len: {len(append)}"
            # )
        else:
            if "m" in freq:
                freq = util.convert_timeframe_to_grouper_compatible(freq)
            group_df = util.group_by(data_slice, freq)
            append = group_df.tail(20)
            tf_series.append(append["close"])
            # print(
            #     f"[{pd.to_datetime(list_dates[save_idx])}] {freq} - "
            #     f'{append.iloc[0]["date"]} to {append.iloc[-1]["date"]}'
            #     f" | len: {len(append)}"
            # )
            assert len(tf_series[-1]) == 20, (
                f"tf_series for {freq} is not 20\n"
                f"tf_series len: {len(tf_series[-1])}\n"
                f"group_df tail: {group_df.tail()}\n"
            )

        # gafs.append(group_dt['close'])
    return tf_series


def create_tf_series_list2(
    data_slices: list[pd.DataFrame], tail_window: int
) -> list[pd.Series]:
    """
    Takes an ohlc within a certain timerange and returns a list of Series containing the last 20
    candles from each timeframe.

    :return: A list of series, each series is the close price for a given timeframe.
    """
    return [ds["close"].tail(tail_window) for ds in data_slices]


def trading_action_new(future_open: float, future_close: float) -> str:
    """
    Given the open and close price of a future, return the trading action

    :param future_open: The open price of the future price bar
    :param future_close: The close price of the future at the time of prediction
    :return: The trading action to take.
    """
    return "LONG" if future_close - future_open > 0 else "SHORT"
