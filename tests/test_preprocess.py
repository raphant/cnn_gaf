from lazyft.data_loader import load_pair_data

from src import models

# region Init
profile = models.Profile(
    pair="BTC/USDT",
    train_timerange="20170102-20211231",
    test_timerange="20220101-",
    timeframes=["30m", "1h", "4h", "1d"],
    image_size=40,
    # download_interval='15m',
)
profile.ensure_directories_exist()

df = load_pair_data("BTC/USDT", timeframe="1h", timerange=profile.train_timerange)
# endregion


def test_preprocess_train():
    pass


def test_generate_decisions():
    decisions = create_decision_map(
        df,
        util.get_dates_from_df(df, profile.max_tf),
        profile.timeframes,
        profile.window_size,
        profile.download_interval,
    )
    assert isinstance(decisions, dict)
    assert isinstance(list(decisions.values())[0], dict)
