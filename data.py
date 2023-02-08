from lazyft.data_loader import load_pair_data


def load_pair_data_for_each_timeframe(pair: str, timerange: str, timeframes: list[str]):
    """
    Loads the data for a given pair, for each timeframe in the given list of timeframes, for the given
    timerange

    :param pair: The pair you want to load data for
    :param timerange: the range of time to load data for
    :param timeframes: list of timeframes to load data for
    :return: A list of dataframes
    """
    return {timeframe: load_pair_data(pair, timeframe, timerange=timerange) for timeframe in timeframes}
