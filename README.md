# cnn_gaf
I also attempt to implement a GAF-CNN based on this paper: [Encoding Candlesticks as Images for Pattern Classification Using Convolutional Neural Networks](https://arxiv.org/pdf/1901.05237.pdf)

## Excerpt
> Candlestick charts display the high, low, opening, and closing prices in a specific period. Candlestick
patterns emerge because human actions and reactions are patterned and continuously replicate. These
patterns capture information on the candles. According to Thomas Bulkowski’s Encyclopedia of
Candlestick Charts, there are 103 candlestick patterns. Traders use these patterns to determine when
to enter and exit. Candlestick pattern classification approaches take the hard work out of visually
identifying these patterns. To highlight its capabilities, we propose a two-steps approach to recognize
candlestick patterns automatically. The first step uses the Gramian Angular Field (GAF) to encode
the time series as different types of images. The second step uses the Convolutional Neural Network
(CNN) with the GAF images to learn eight critical kinds of candlestick patterns. In this paper, we call
the approach GAF-CNN. In the experiments, our approach can identify the eight types of candlestick
patterns with 90.7% average accuracy automatically in real-world data, outperforming the LSTM
model.



## Notebooks
`cnn.ipynb` - An example of the work flow

`cnn_lstm.ipynb`- Here I implement a LSTM layer into the model

`tuner.ipynb` - An implementation of parameter hyperopting for the model.

[LazyFT](https://github.com/raphant/lazyft) is used to download data and test strategies. 
