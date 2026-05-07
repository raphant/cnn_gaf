# cnn_gaf

A research-oriented project for transforming market time series into **Gramian Angular Field (GAF)** images and training **Convolutional Neural Networks (CNNs)** to classify trading direction from those images.

This repository explores the idea from the paper:

> [Encoding Candlesticks as Images for Pattern Classification Using Convolutional Neural Networks](https://arxiv.org/pdf/1901.05237.pdf)

In this implementation, OHLC price data is converted into image representations across multiple timeframes, then used to train CNN-based models for **LONG/SHORT classification**. The repo also includes experiments with **CNN+LSTM** and **hyperparameter tuning**.

---

## Overview

The project pipeline is roughly:

1. **Load historical OHLC market data**
2. **Slice data into rolling windows across multiple timeframes**
3. **Convert each window into GAF image representations**
4. **Label samples as LONG or SHORT**
5. **Train a CNN model on the generated image dataset**
6. **Evaluate and export trained models**

This repo is best understood as an experimental ML workflow for financial time series classification rather than a polished end-user package.

---

## Features

- Convert candlestick / OHLC time series into **GAF-based images**
- Build training datasets from **multiple timeframes**
- Train CNN models for **LONG vs SHORT classification**
- Experiment with **CNN + LSTM**
- Run **hyperparameter tuning** with Keras Tuner / AutoKeras
- Save trained models and run predictions on new data

---

## Repository Structure

- `preprocessing.py` — data preprocessing, timeframe splitting, decision mapping, image generation
- `train.py` — model training, generator creation, evaluation helpers, tuning entry points
- `predict.py` — inference utilities for prediction on new data
- `cnn_model.py` — CNN model definitions and model factory functions
- `split.py` — logic for splitting time series into windows / timeframes
- `writer.py` — writing generated GAF images
- `data.py` — data loading helpers
- `data_models.py` — core profile/mode models
- `util.py` — utility functions
- `constants.py` — project constants and path definitions
- `cnn.ipynb` — notebook showing an example end-to-end workflow
- `cnn_lstm.ipynb` — notebook exploring a CNN + LSTM architecture
- `tuner.ipynb` — notebook for hyperparameter optimization experiments
- `backtest.py` — backtesting-related experimentation
- `tests/` — test files
- `exports/` — generated model artifacts
- `user_data/` — local runtime data and downloaded market data

---

## Requirements

This project is Python-based and depends on ML / data libraries including:

- pandas
- numpy
- scipy
- keras / tensorflow
- scikit-learn
- pillow
- pyts
- matplotlib
- diskcache
- loguru
- pydantic
- autokeras

It also references **local editable dependencies** in `requirements.txt`:

- `../ftworkdir/lazyft_pkg/`
- `../freqtrade/`

The project currently assumes a **local research environment** where those repositories are already cloned and available at the relative paths above.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/raphant/cnn_gaf.git
cd cnn_gaf
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installation may fail unless the local editable dependencies referenced in `requirements.txt` (`../ftworkdir/lazyft_pkg/` and `../freqtrade/`) exist on your machine. This project is currently set up for a local research environment and does not yet provide a fully self-contained installation.

[LazyFT](https://github.com/raphant/lazyft) is used as part of the broader workflow for downloading data and testing strategies.

---

## Usage

### Run training

```bash
python train.py
```

This uses the default configuration in `train.py`, which currently trains on `BTC/USDT` with the following defaults:

- image size: `40`
- training timerange: `20170102-20211231`
- testing timerange: `20220101-`
- timeframes: `["1h", "4h", "12h", "1d"]`

### Explore notebooks

The notebooks are the easiest place to understand the project end-to-end:

- `cnn.ipynb` — example workflow
- `cnn_lstm.ipynb` — CNN + LSTM architecture experiments
- `tuner.ipynb` — hyperparameter optimization

---

## Disclaimer

This project is for **research and experimentation only**.  
It should **not** be considered financial advice or a production-ready trading system.

---

## License

This project is licensed under the [MIT License](LICENSE).
