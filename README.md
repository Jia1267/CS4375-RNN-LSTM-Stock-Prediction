# CS4375 RNN/LSTM Stock Price Prediction

## Project Overview

This project is for CS 4375 Machine Learning. The goal of this project is to use historical stock market data to predict the next day's closing price. We use Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) models to compare how well each model can learn time series patterns.

The main prediction task is:

> Use previous stock market data to predict the next trading day's closing price.

For the RNN model, the main model logic is implemented from scratch using NumPy instead of using built-in deep learning models from TensorFlow, Keras, or PyTorch.

---

## Dataset

The dataset used in this project is the Yahoo Finance Dataset from Kaggle:

https://www.kaggle.com/datasets/suruchiarora/yahoo-finance-dataset-2018-2023

The dataset contains daily stock market records from 2018 to 2023.

The main columns used are:

- Date
- Open
- High
- Low
- Close*
- Adj Close**
- Volume

The target column is: Close*
This means the model predicts the next trading day's closing price.

```text
Project Structure
CS4375-RNN-LSTM-Stock-Prediction/
│
├── data/
│   └── stock_data.xlsx
│
├── RNN/
│   ├── rnn_stock_prediction.py
│   ├── training_loss.png
│   └── rnn_prediction_result.png
│
├── LSTM/
│   └── lstm_stock_prediction.py
│
|
|
├── Report
|   └── CS 4375 Project Report.pdf
|
├── README.md
└── requirements.txt


RNN Model
The RNN model is implemented manually using NumPy. The code includes the forward pass, backward pass, gradient clipping, and parameter updates.
The model uses previous stock data as a sequence input and predicts the next closing price. In the best RNN experiment, the model uses the previous 20 trading days to predict the next trading day's Close* value.


Best RNN Result
After testing different hyperparameters, the best RNN result used the following setting:
Model: Simple RNN
Target column: Close*
Sequence length: 20
Hidden size: 32
Learning rate: 0.01
Epochs: 50
Loss function: Mean Squared Error (MSE)
Optimizer: Stochastic Gradient Descent (SGD)
Train/Test Split: 80/20
Dataset size: 1258

Final test result:
Test MSE: 154929.8222
Test RMSE: 393.6113
Test MAE: 303.2032


How to Run
Install the required packages:
pip install numpy pandas matplotlib scikit-learn openpyxl


Output
Running the RNN code will generate result figures such as:

training_loss.png
rnn_prediction_result.png

The training loss figure shows how the loss changes during training.
The prediction result figure compares the actual closing price with the predicted closing price.


Evaluation Metrics
The model is evaluated using:
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Lower values mean better prediction performance.


