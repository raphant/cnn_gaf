import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable

import numpy as np
import pandas as pd
import PIL
from diskcache import Cache
from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
from lazyft import paths
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL.Image import Image
from pyts.image import GramianAngularField

from constants import alive_bar

figure_cache = Cache(paths.CACHE_DIR / "figures")
image_creator_cache = Cache(paths.CACHE_DIR / "img_creator")
data_gen = ImageDataGenerator(rescale=1 / 255)


def fill_writer_queue_with_decisions(image_save_path: Path, images_data: dict):
    """
    This function takes a path to save images to and a dictionary of images to save, and returns a
    queue of images to save

    :param image_save_path: The path to the folder where you want to save the images
    :param images_data: A dictionary of decision: (datestamp, quad_series) pairs
    :return: A queue of dictionaries.
    """
    queue = Queue()
    image_save_path.joinpath("LONG").mkdir(exist_ok=True)
    image_save_path.joinpath("SHORT").mkdir(exist_ok=True)
    pngs = [p.name for p in Path(image_save_path).glob("*/*.png")]
    for decision, data in images_data.items():
        logger.info(f"Populating {decision} image queue...")
        for datestamp, quad_series in data:
            save_name = str(Path(datestamp.replace("-", "_")).with_suffix(".png"))
            if save_name in pngs:
                continue
            queue.put(
                {
                    "x_plots": [create_gaf(x)["gadf"] for x in quad_series],
                    "image_name": save_name,
                    "destination": decision,
                    "folder": image_save_path,
                }
            )
    return queue


def fill_writer_queue(data: dict[str, list[pd.Series]], folder: Path):
    """
    This function takes a dictionary of data and a folder, and returns a queue of data to be plotted
    and a list of threads.

    :param data: a dictionary of dataframes, where the keys are the datestamps of the dataframes
    :param folder: the folder where the images will be saved
    :return: A queue and a list of threads.
    """
    logger.info(f"Creating writer queue... | Save folder: {folder}")
    queue = Queue()
    pngs = [p.stem for p in folder.glob("*.png")]
    for datestamp, d in data.items():
        if datestamp in pngs:
            continue
        to_plot = [create_gaf(x)["gadf"] for x in d]
        queue.put({"x_plots": to_plot, "image_name": datestamp, "folder": folder})
    return queue


def create_gaf_images(data: dict[str, list[pd.Series]]):
    """
    This function takes a dictionary of data and a folder, and returns a queue of data to be plotted
    and a list of threads.

    :param data: a dictionary of dataframes, where the keys are the datestamps of the dataframes
    :param folder: the folder where the images will be saved
    :return: A queue and a list of threads.
    """
    gafs = {}
    with alive_bar(len(data), title="Creating GAF images...") as bar:
        for datestamp, d in data.items():
            standardize = calculate_standardized_gaf(d)
            gafs[datestamp] = standardize.reshape(1, 40, 40, 3)
            bar()
    return gafs


@image_creator_cache.memoize()
def calculate_standardized_gaf(d):
    """
    It takes a list of numpy arrays, creates a GAF for each array, then standardizes the GAFs

    :param d: a list of numpy arrays
    :return: A standardized GAF
    """
    to_plot = [create_gaf(x)["gadf"] for x in d]
    figure = create_figure(to_plot)
    figure_to_numpy = numpy_from_figure(figure)
    return data_gen.standardize(figure_to_numpy)


def numpy_from_figure(fig: Figure) -> np.ndarray:
    """
    It takes a matplotlib figure, draws it, converts it to a PIL image, resizes it, and converts it to a
    numpy array

    :param fig: The figure to convert to a numpy array
    :return: A numpy array of the image
    """
    fig.canvas.draw()
    # noinspection PyNoneFunctionAssignment
    image = PIL.Image.frombuffer(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    image = image.resize((40, 40))
    plt.close(fig)
    return img_to_array(image)


def write_images(writer_queue: Queue) -> None:
    """
    It takes 4 list of close price series and creates a GADF image from each of them

    :param writer_queue: A queue of dictionaries containing the data to create the image
    :return: A PIL image object
    """
    if writer_queue.empty():
        return
    threads = []
    event = Event()
    with alive_bar(writer_queue.qsize(), title="Creating images...") as bar:
        for i in range(2):
            t = Thread(
                target=image_writer,
                args=(writer_queue, event, i, bar, create_images),
            )
            threads.append(t)
            t.start()
        try:
            writer_queue.join()
        except KeyboardInterrupt:
            event.set()
            raise
    event.set()


def image_writer(
    queue: Queue, event: Event, index: int, bar: Callable, func: Callable
) -> None:
    """
    It takes a queue as input, and it reads the queue for data. If it finds data, it creates an image

    :param queue: A queue to read data from
    :param event: An event to stop the thread
    :param index: The index of the thread
    :param bar: A progress bar to update
    :param func: A function to create the image
    """
    logger.info(f"Starting image writer {index}...")
    while not event.is_set():
        try:
            data = queue.get(block=False)
        except Empty:
            break

        try:
            func(**data)
        except Exception as e:
            logger.exception(f"Error in image_writer {index}: {e}")
            queue.put(data)
            continue
        queue.task_done()
        bar()
    logger.info(f"Image writer {index} stopped.")


def create_images(
    x_plots: Any,
    image_name: str,
    destination: str = "",
    folder=None,
) -> None:
    """
    Create a grid of images and save them to disk

    :param x_plots: The list of images to be plotted
    :param image_name: The name of the image
    :param destination: The name of the folder where the images will be saved
    :param image_matrix: tuple = (2, 2)
    :param folder: The folder where the images will be saved
    :return: The path to the image.
    """
    t1 = time.perf_counter()
    try:
        fig = create_figure(x_plots)
        save_path = folder / destination / image_name
        # save_path.parent.mkdir(exist_ok=True, parents=True)
        # t3 = time.perf_counter()
        # exists = save_path.exists()
        # print(
        #     f'Elapsed time -> exists [{exists}]:', timedelta(seconds=time.perf_counter() - t3)
        # )
        fig.savefig(save_path)
        plt.close(fig)
    except Exception:
        raise
    finally:
        pass
        # print('create_images() Elapsed time:', timedelta(seconds=time.perf_counter() - t1))


# @figure_cache.memoize()
def create_figure(x_plots, image_matrix=(2, 2)):
    """
    It takes a matrix of images and plots them in a grid

    :param image_matrix: The number of rows and columns of images to display
    :param x_plots: a list of images to plot
    :return: A figure object
    """
    fig: Figure = plt.figure(figsize=[img * 4 for img in image_matrix])
    grid = ImageGrid(
        fig,
        111,
        axes_pad=0,
        nrows_ncols=image_matrix,
        share_all=True,
    )
    images = x_plots
    for image, ax in zip(images, grid):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap="rainbow", origin="lower")
    return fig


def create_gaf(ts: pd.Series) -> dict[str, np.ndarray]:
    """
    Create a dictionary with a key 'gadf' and a value that is the output of the GramianAngularField
    function

    :param ts: The time series to be converted to GADF
    :return: A dictionary with the key 'gadf' and the value of the GADF.
    """
    gadf = GramianAngularField(method="difference", image_size=ts.shape[0])
    return {"gadf": gadf.fit_transform(pd.DataFrame(ts).T)[0]}
