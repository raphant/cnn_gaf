from functools import partial
from pathlib import Path

from alive_progress import alive_bar

PATH = Path(__file__).parent
IMAGES_PATH = Path(PATH, 'gaf_images')
DATA_PATH = Path(PATH, 'TimeSeries')
REPO = PATH / 'Models'
PATH_DOC = PATH / 'Documents'
PATH_OUT = PATH / 'Output'
DEFAULT_TIMEFRAMES = ['1h', '4h', '8h', '12h']
alive_bar = partial(alive_bar, force_tty=True)
