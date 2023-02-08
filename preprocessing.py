import gc
import time
from pathlib import Path
from typing import Union

import memory_profiler
import mpu
import pandas as pd
from diskcache import Cache
from keras_preprocessing.image import ImageDataGenerator
from lazyft import paths
from lazyft.data_loader import load_pair_data
from loguru import logger
from pandas import DataFrame, Series

import util
from data import load_pair_data_for_each_timeframe
from data_models import Mode, Profile
from split import (
    create_decision_map,
    create_decision_map_with_custom_timeframes,
    split_timeframes,
    split_timeframes2,
)
from writer import (
    create_gaf_images,
    fill_writer_queue,
    fill_writer_queue_with_decisions,
    write_images,
)

cache = Cache(paths.CACHE_DIR)
data_gen = ImageDataGenerator(rescale=1 / 255)


def preprocess(
    profile: Profile,
    mode: Mode,
    custom_data: pd.DataFrame = None,
    custom_timeframes: dict[str, pd.DataFrame] = None,
    save_preprocessed_file: bool = True,
    load_preprocessed_file: bool = True,
):
    """
    It takes the dataframe of the pair

    :param profile: The profile of the session
    :param mode: Mode = TRAIN, TEST, or PREDICT
    :param custom_data: A custom dataframe to use for preprocessing
    :param custom_timeframes: A dictionary of custom timeframes to use for grouping the data
    :param load_preprocessed_file: If true, load the preprocessed file from disk
    :param save_preprocessed_file: If true, save the preprocessed file to disk
    :return: a dataframe with the following columns:
    """
    if mode == Mode.PREDICT and custom_data is None:
        raise ValueError("custom_data must be specified for prediction mode")

    df = custom_data
    if df is None:
        df = load_pair_data(
            profile.pair,
            profile.download_interval,
            timerange=profile.get_timerange(mode),
        )

    df = df[["date", "close", "open"]]
    # clean_df = clean_non_trading_times(df)
    try:
        if mode == Mode.PREDICT:
            return preprocess_prediction(custom_timeframes, df, profile)

        return preprocess_train(df, load_preprocessed_file, mode, profile, save_preprocessed_file)
    except Exception as e:
        raise
    finally:
        gc.collect()


def preprocess2(
    profile: Profile,
    mode: Mode,
    custom_data: dict[str, pd.DataFrame] = None,
    # custom_timeframes: list[pd.DataFrame] = None,
    save_preprocessed_file: bool = True,
    load_preprocessed_file: bool = True,
):
    """
    It takes the dataframe of the pair

    :param profile: The profile of the session
    :param mode: Mode = TRAIN, TEST, or PREDICT
    :param custom_data: A custom dataframe to use for preprocessing
    :param custom_timeframes: A dictionary of custom timeframes to use for grouping the data
    :param load_preprocessed_file: If true, load the preprocessed file from disk
    :param save_preprocessed_file: If true, save the preprocessed file to disk
    :return: a dataframe with the following columns:
    """
    if mode == Mode.PREDICT and custom_data is None:
        raise ValueError("custom_data must be specified for prediction mode")

    dfs = custom_data
    if dfs is None:
        dfs = load_pair_data_for_each_timeframe(
            profile.pair,
            timerange=profile.get_timerange(mode),
            timeframes=profile.timeframes,
        )
    for key, value in dfs.items():
        dfs[key].loc[:, "date"] = pd.to_datetime(value["date"])
        dfs[key] = value[["date", "close", "open"]]
    # clean_df = clean_non_trading_times(df)
    try:
        if mode == Mode.PREDICT:
            return preprocess_prediction2(dfs, profile)

        return preprocess_train2(dfs, mode, profile, save_preprocessed_file)
    except Exception as e:
        raise
    finally:
        gc.collect()


def preprocess_train(df, load_preprocessed_file, mode, profile, save_preprocessed_file):
    """
    Preprocessing for training and evaluation

    :param df: the dataframe containing the data
    :param load_preprocessed_file: If you've already preprocessed the data, you can load it from a file
    :param mode: "train" or "test"
    :param profile: the name of the profile you want to use
    :param save_preprocessed_file: If True, the preprocessed data will be saved to a file
    :return: A list of all the images that were written to disk.
    """
    df["date"] = pd.to_datetime(df["date"])
    decisions, image_dir = map_date_slices_to_decision(
        df, profile, mode, save_preprocessed_file, load_preprocessed_file
    )
    write_images(fill_writer_queue_with_decisions(image_dir, decisions))
    return [
        *[d[0] for d in decisions["SHORT"]],
        *[d[0] for d in decisions["LONG"]],
    ]


def preprocess_train2(
    dfs: dict[str, pd.DataFrame],
    mode: Mode,
    profile: Profile,
    load_preprocessed_file=False,
    save_preprocessed_file=False,
):
    """
    Preprocessing for training and evaluation

    :param dfs: the dataframe containing the data
    :param load_preprocessed_file: If you've already preprocessed the data, you can load it from a file
    :param mode: "train" or "test"
    :param profile: the name of the profile you want to use
    :param save_preprocessed_file: If True, the preprocessed data will be saved to a file
    :return: A list of all the images that were written to disk.
    """
    # df["date"] = pd.to_datetime(df["date"])
    decisions, image_dir = map_date_slices_to_decision2(
        dfs, profile, mode, save_preprocessed_file, load_preprocessed_file
    )
    write_images(fill_writer_queue_with_decisions(image_dir, decisions))
    return [
        *[d[0] for d in decisions["SHORT"]],
        *[d[0] for d in decisions["LONG"]],
    ]


def preprocess_prediction(
    custom_timeframes: dict[str, pd.DataFrame], df: pd.DataFrame, profile: Profile
):
    """
    Preprocessing for freqtrade strategies

    :param custom_timeframes: a list of timeframes to use for prediction. If you want to use all the
    timeframes in the profile, leave this as None
    :param df: the dataframe of the data you want to predict on
    :param profile: the profile object that contains all the parameters for the model
    :return: The parent folder of the temporary folder and a list of the keys of the quad_series
    dictionary.
    """
    quad_series = split_timeframes(
        df,
        profile.window_size,
        profile.timeframes,
        util.get_dates_from_df(df, profile.max_tf),
        download_interval=profile.download_interval,
        custom_timeframes=custom_timeframes,
    )
    start_date = df.iloc[0]["date"]
    end_date = df.iloc[-1]["date"]
    tmp_folder = Path(
        "/tmp",
        profile.formatted_pair,
        f"{int(start_date.timestamp())}-{int(end_date.timestamp())}",
        "_".join(profile.timeframes),
    )
    tmp_folder.mkdir(exist_ok=True, parents=True)
    # write_images(fill_writer_queue(quad_series, tmp_folder))
    # return tmp_folder.parent, list(quad_series.keys())

    return create_gaf_images(quad_series)


def preprocess_prediction2(dfs: dict[str, pd.DataFrame], profile: Profile):
    """
    Preprocessing for freqtrade strategies
    """
    main_tf_data = dfs[profile.max_tf]["date"]
    quad_series = split_timeframes2(
        dfs,
        profile.window_size,
        util.get_dates_from_df2(main_tf_data),
        profile.tail_window,
    )
    start_date = main_tf_data.iloc[0]["date"]
    end_date = main_tf_data.iloc[-1]["date"]
    tmp_folder = Path(
        "/tmp",
        profile.formatted_pair,
        f"{int(start_date.timestamp())}-{int(end_date.timestamp())}",
        "_".join(profile.timeframes),
    )
    tmp_folder.mkdir(exist_ok=True, parents=True)
    # write_images(fill_writer_queue(quad_series, tmp_folder))
    # return tmp_folder.parent, list(quad_series.keys())

    return create_gaf_images(quad_series)


def map_date_slices_to_decision(df, profile, mode, save_preprocessed_file, load_preprocessed_file):
    """
    > It creates a decision map for the given dataframe and mode, and returns the decision map and the
    image directory

    :param df: the dataframe containing the data
    :param profile: The profile object that contains all the parameters for the model
    :param mode: This is the mode of the data. It can be either train, test or validation
    :param save_preprocessed_file: If True, the preprocessed data will be saved to a pickle file
    :param load_preprocessed_file: If True, the preprocessed file will be loaded from the image
    directory
    :return: A decision map and the image directory.
    """
    image_dir = profile.get_images_path(mode)
    list_dates = util.get_dates_from_df(df, profile.max_tf)
    decision_map = None
    pickle_path = Path(image_dir, f"preprocessed_{mode.value}.pickle")
    if load_preprocessed_file:
        if not pickle_path.exists():
            logger.info(f"Preprocessed file not found at {pickle_path}, preprocessing data")
        else:
            logger.info(f"Loading preprocessed file from {pickle_path}")
            decision_map = mpu.io.read(str(pickle_path))
    if decision_map is None:
        logger.info(f"Creating decision map for {mode.value} mode...")
        decision_map = create_decision_map_with_custom_timeframes(
            df,
            list_dates,
            profile.timeframes,
            profile.window_size,
            profile.download_interval,
        )
        if save_preprocessed_file:
            logger.info(f"Saving preprocessed file to {pickle_path}")
            mpu.io.write(str(pickle_path), decision_map)
    util.print_preprocess_info(decision_map, len(list_dates))
    logger.info(f"Creating images for {mode.value} mode......")
    return decision_map, image_dir


def map_date_slices_to_decision2(
    dfs: dict[str, pd.DataFrame],
    profile,
    mode,
    save_preprocessed_file=False,
    load_preprocessed_file=False,
):
    """
    > It creates a decision map for the given dataframe and mode, and returns the decision map and the
    image directory

    :param df: the dataframe containing the data
    :param profile: The profile object that contains all the parameters for the model
    :param mode: This is the mode of the data. It can be either train, test or validation
    :param save_preprocessed_file: If True, the preprocessed data will be saved to a pickle file
    :param load_preprocessed_file: If True, the preprocessed file will be loaded from the image
    directory
    :return: A decision map and the image directory.
    """
    image_dir = profile.get_images_path(mode)
    list_dates = util.get_dates_from_df2(dfs[profile.max_tf])
    decision_map = None
    pickle_path = Path(image_dir, f"preprocessed_{mode.value}.pickle")
    if load_preprocessed_file:
        if not pickle_path.exists():
            logger.info(f"Preprocessed file not found at {pickle_path}, preprocessing data")
        else:
            logger.info(f"Loading preprocessed file from {pickle_path}")
            decision_map = mpu.io.read(str(pickle_path))
    if decision_map is None:
        logger.info(f"Creating decision map for {mode.value} mode...")
        decision_map = create_decision_map_with_custom_timeframes(
            dfs, profile.max_tf, list_dates, profile.window_size, profile.tail_window
        )
        if save_preprocessed_file:
            logger.info(f"Saving preprocessed file to {pickle_path}")
            mpu.io.write(str(pickle_path), decision_map)
    util.print_preprocess_info(decision_map, len(list_dates))
    logger.info(f"Creating images for {mode.value} mode......")
    return decision_map, image_dir


# @util.timeit
# def quick_gaf(df: pd.DataFrame, profile: Profile):
#     """
#     The function takes a dataframe of OHLCV data and returns a list of GAFs and a list of answers
#
#     :param df: The dataframe to be processed
#     :param profile: The profile to use for the data
#     :return: A list of GAFs and a list of answers.
#     """
#     # t1 = time.perf_counter()
#
#     quad_series = split_timeframes(df, timeframes=profile.timeframes)
#     to_plot = [create_gaf(x)["gadf"] for x in quad_series]
#     image = get_image_from_gaf(to_plot)
#     # print('quick_gaf() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
#     image = image.resize((profile.image_size, profile.image_size))
#
#     as_numpy = img_to_array(image)
#     as_numpy = as_numpy.reshape((1, profile.image_size, profile.image_size, 3))
#     return data_gen.standardize(as_numpy)


# def data_to_image_preprocess_new(
#     timerange='20180101-20211231',
#     data: pd.DataFrame = None,
#     pair: str = 'BTC/USDT',
#     download_interval: str = '1h',
#     high_interval: str = '1d',
#     image_save_path: Path = TRAIN_IMAGES_PATH,
# ):
#     """
#     This function takes a timerange and a dataframe and creates images for each timeframe.
#     If no dataframe is passed, it will load the dataframe from the lazyft data.
#
#     :param timerange: The time range to use for the data, defaults to 20180101-20211231 (optional)
#     :param data: The dataframe to be converted to an image
#     :param pair: The coin to be used for the data, defaults to 'BTC/USDT' (optional)
#     :param download_interval: The interval to be used for downloading data, defaults to '1h' (optional)
#     :param high_interval: The maximum timeframe to use. This will be used to create the labels.
#     :param image_save_path: The path to save the images, defaults to IMAGES_PATH (optional)
#     :type data: pd.DataFrame
#     :return: A dataframe with the following columns:
#         date, open, close, high, low, volume
#     """
#     dl_tf_mins = timeframe_to_minutes(download_interval)
#     high_tf_mins = timeframe_to_minutes(high_interval)
#
#     assert dl_tf_mins < high_tf_mins, 'interval must be less than high_interval'
#
#     data_gen = quadrant_data_generator(pair, ['1h', '4h', '8h'], timerange)
#     # clean_df = clean_non_trading_times(df)
#     return set_gaf_data_new(data_gen, image_save_path=image_save_path, high_interval=high_interval)
#
#
# def set_gaf_data_new(
#     df: pd.DataFrame,
#     window: int = 20,
#     timeframes: list = None,
#     image_save_path=TRAIN_IMAGES_PATH,
#     high_interval='1d',
# ):
#     """
#     It takes a dataframe of historical data and a window size,
#     and generates a set of images for each trading decision (long or short) that occurred during that
#     window
#
#     :param df: The dataframe that contains the data
#     :param window: The number of days to look back when calculating the GAF, defaults to 20
#     :param timeframes: list of timeframes to use for the GAF
#     :param image_save_path: The path to save the images, defaults to IMAGES_PATH (optional)
#     :param high_interval: The maximum timeframe to use. This will be used to create the labels.
#     """
#     if timeframes is None:
#         timeframes = ['1h', '4h', '8h', '1d']
#
#     dates = df['date'].dt.date
#     dates = dates.drop_duplicates()
#     list_dates = dates.apply(str).tolist()
#
#     index = window
#     decision_map = defaultdict(list)
#     df_grouped_1d = df.groupby(pd.Grouper(key='date', freq='1d')).mean().reset_index()
#     # create new Series from df, with the 'SHORT' being set if CLOSE-OPEN < 0 and 'LONG' otherwise
#     df_grouped_1d['decision'] = df_grouped_1d['close'].shift(-1) - df_grouped_1d['open'].shift(-1)
#     decisions = df_grouped_1d['decision'] = (
#         df_grouped_1d['decision'].apply(lambda x: 'SHORT' if x < 0 else 'LONG').to_list()
#     )
#     # del df_grouped_1d
#     for index, date in enumerate(list_dates[window:], start=window):
#         if index >= len(list_dates):
#             break
#
#         # select appropriate timeframe
#         data_slice = df.loc[
#             (df['date'] > list_dates[index - window]) & (df['date'] < list_dates[index])
#         ]
#         # print('len of data slice: ', len(data_slice), 'head: ', data_slice.head())
#         gafs = []
#
#         # group dataslices by timeframe
#         for freq in timeframes:
#             group_dt = data_slice.groupby(pd.Grouper(key='date', freq=freq)).mean().reset_index()
#             group_dt = group_dt.dropna()
#             gafs.append(group_dt['close'].tail(20))
#         future_value = df.loc[df['date'] == date]['close'].iloc[-1]
#         future_open = df.loc[df['date'] == date]['open'].iloc[-1]
#
#         current_value = data_slice['close'].iloc[-1]
#         decision = trading_action(future_value, current_value)
#         new_decision = decisions[index - 1]
#         try:
#             assert decision == new_decision
#         except:
#             print(
#                 'list_dates[index]: ',
#                 list_dates[index],
#                 'original decision: ',
#                 decision,
#                 'New decision: ',
#                 new_decision,
#                 'index: ',
#                 index,
#                 'future_close: ',
#                 future_value,
#                 'future_open: ',
#                 future_open,
#                 'current_value: ',
#                 current_value,
#                 # sep=' | ',
#             )
#             raise
#         decision_map[decision].append([list_dates[index - 1], gafs])
#         index += 1
#     exit()
#     print('Generating images...')
#     generate_gaf(decision_map, image_save_path)
#     dt_points = dates.shape[0]
#     total_shorts = len(decision_map['SHORT'])
#     total_longs = len(decision_map['LONG'])
#     images_created = total_shorts + total_longs
#     print(
#         "========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}"
#         "\nTotal LONG positions: {2}\nTotal SHORT positions: {3}".format(
#             dt_points, images_created, total_shorts, total_longs
#         )
#     )


# def generate_gaf_pooled(images_data: dict[str, list], image_save_path=TRAIN_IMAGES_PATH):
#     """
#     A multithreaded version of generate_gaf
#     """
#
#     # in_memory_mode = image_save_path != TRAIN_IMAGES_PATH
#     pngs = [p.name for p in Path(image_save_path).glob(f"*/*.png")]
#
#     def func(data, decision):
#         save_name = str(Path(data[0].replace('-', '_')).with_suffix('.png'))
#         if save_name in pngs:
#             return
#         t1 = time.time()
#         first = data[1]
#         to_plot = [create_gaf(x)['gadf'] for x in first]
#         create_images(
#             x_plots=to_plot,
#             image_name=save_name.replace('.png', ''),
#             destination=decision,
#             folder=image_save_path,
#         )
#         print('generate_gaf_pooled() -> func() Elapsed time:', timedelta(seconds=time.time() - t1))
#
#     # images = []
#     # decisions = []
#     with ThreadPoolExecutor() as executor:
#         for decision, data in images_data.items():
#             futures = [executor.submit(func, data_point, decision) for data_point in data]
#             for future in as_completed(futures):
#                 future.result()
#                 # images.append(image)
#                 # decisions.append(1 if decision == 'LONG' else 0)
#     #
#     # if in_memory_mode:
#     #     return images, decisions


def create_dataflow_dataframe(path: Path, list_dates: list) -> Union[DataFrame, Series]:
    """
    :param path: As String
    :param list_dates:
    :return: List of overlapping index DataFrames
    """
    all_images = path.glob("*/*.png")
    df_images = []
    labels = []
    for i in sorted(all_images, key=lambda x: x.name):
        if i.stem not in list_dates:
            continue
        sub_folder = i.parent.name
        labels.append(sub_folder)
        df_images.append(str(i.resolve()))

    df = pd.DataFrame(
        {
            "Images": df_images,
            "Labels": labels,
            "Dates": sorted(list_dates),
        }
    )
    df["Dates"] = pd.to_datetime(df["Dates"])
    # dataframes.append(data_slice)
    # data = pd.concat(dataframes)
    df.sort_values(by="Dates", inplace=True)
    # del df["Dates"]
    return df


# def get_gaf_and_answers(profile: Profile, mode=Mode.TRAIN):
#     """
#     It loads the data, groups it by 480 rows, and then creates a GAF image for each group
#
#     :param profile: The Profile object that contains the parameters for the GAF
#     :param mode: The mode of the data
#     :return: The answers and the images
#     """
#     data = load_pair_data(
#         profile.pair, profile.download_interval, timerange=profile.get_timerange(mode)
#     )
#     # validation_data = load_pair_data(profile.pair, '1h', timerange='20220101-')
#     # gb = data.groupby(np.arange(len(data)) // (profile.max_tf_as_hours * 20))
#     gb = util.roll(data, profile.max_tf_as_hours * 20)
#     images = np.zeros(shape=(len(gb), profile.image_size, profile.image_size, 3))
#     answers = np.zeros(shape=(len(gb), 1), dtype=int)
#     with alive_bar(
#         len(data) // (profile.max_tf_as_hours * 20),
#         title="Creating Training Data...",
#         bar="smooth",
#     ) as bar:
#         for i, d in gb:
#             image = quick_gaf(d, profile)
#             decision = get_answer(d, data)
#             images[i] = image
#             answers[i] = 1 if decision == "SHORT" else 0
#         bar()
#
#     return answers, images


if __name__ == "__main__":
    tmp = Path("/tmp/test_btc_2022")
    tr = "20220314-"
    # threaded = False
    # t1 = time.time()
    # data_to_image_preprocess(timerange=tr)
    # print('Non MultiThreaded Elapsed time:', timedelta(seconds=time.time() - t1))
    threaded = False
    t2 = time.time()
    profile = Profile(pair="BTC/USDT", train_timerange=tr, timeframes=["1h", "4h", "12h", "1d"])
    # profile.ensure_directories_exist()
    data = load_pair_data(profile.pair, "1h", timerange=tr)
    # data_to_image_preprocess(profile)
    # print('MultiThreaded Elapsed time:', timedelta(seconds=time.time() - t2))
    # preprocess = data_to_image_preprocess(
    #     # timerange='20170101-20211231',
    #     data=load_pair_data('BTC/USDT', '30m', timerange='20220223-'),
    #     image_save_path=tmp,
    # )
    # x = preprocess[0]
    # y = preprocess[1]
    # images = quick_gaf(load_pair_data('BTC/USDT', '30m', timerange='20220223-'))
    # images = transform(images)
    # train_dataset = datasets.ImageFolder(
    #     root=str(tmp),
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(255),
    #             transforms.ToTensor()
    #             # transforms.Scale(255),
    #         ]
    #     ),
    # )
    # generator = ImageDataGenerator(rescale=1 / 255)
    # data = generator.flow(x=images, batch_size=1)
    # print(torch_x)
    # print(train_dataset[5][0])
    # generate_gaf(preprocess)
