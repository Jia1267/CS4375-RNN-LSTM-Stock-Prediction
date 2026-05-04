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


LSTM Model
The LSTM model is implemented manually using PyTorch basic tensor operations and custom LSTM gate calculations. The model does not use the built-in PyTorch LSTM layer. Instead, it manually computes the forget gate, input gate, candidate cell state, output gate, cell state, and hidden state.

The model uses previous stock data as a sequence input and predicts the next trading day's Close* value. In the best LSTM experiment, the model uses the previous 20 trading days to predict the next trading day's closing price.


Best LSTM Result
After testing different hyperparameters, the best LSTM result used the following setting:

Model: Manual LSTM
Target column: Close*
Sequence length: 20
Hidden size: 32
Learning rate: 0.01
Epochs: 50
Loss function: Mean Squared Error (MSE)
Optimizer: Adam
Train/Test Split: 80/20
Dataset size: 1258
Train samples: 990
Test samples: 248

Final Training Loss: 0.000687

Final test result:
Test MSE: 252649.4531
Test RMSE: 502.6425
Test MAE: 394.5134
Test MAPE: 1.2303%

How to Run LSTM
Install the required packages:
pip install numpy pandas matplotlib openpyxl torch

Run the LSTM code:
python lstm_stock_prediction.py


LSTM Output
Running the LSTM code will generate result figures such as:

lstm_training_loss.png
lstm_prediction_result.png

The LSTM training loss figure shows how the model loss changes during training.
The LSTM prediction result figure compares the actual closing price with the predicted closing price.


LSTM Experiment Log
The LSTM experiments are saved in:

exp_log.txt

This file records the input settings and output results for each experiment, including sequence length, hidden size, learning rate, epochs, final training loss, MSE, RMSE, MAE, and MAPE.




RNN and LSTM Comparison
Based on the final test results, the RNN model achieved lower MSE, RMSE, and MAE than the LSTM model in this project. The best RNN model had a Test RMSE of 393.6113 and Test MAE of 303.2032, while the best LSTM model had a Test RMSE of 502.6425 and Test MAE of 394.5134.

This means the RNN model performed better on this dataset. One possible reason is that the dataset is relatively small, with only 1258 rows. The LSTM model has a more complex structure, so it may need more data or more tuning to perform better. However, the LSTM model still produced a low MAPE of 1.2303%, which means its average percentage prediction error was small.

Model Comparison Table

Model        MSE           RMSE       MAE        
RNN          154929.8222   393.6113   303.2032   
LSTM         252649.4531   502.6425   394.5134   

From the table, the RNN model has lower MSE, RMSE, and MAE than the LSTM model. Therefore, the RNN model performed better overall in this project.
