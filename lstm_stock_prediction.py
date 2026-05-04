import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


data_file = ["stock_data.xlsx", "yahoo_data.xlsx"]
target_c = "Close*"
sequence_length = 20
t_ratio = 0.80
HIDDEN_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01




def find_data():
   for filename in data_file:
       if os.path.exists(filename):
           return filename
   raise FileNotFoundError("Could not find stock_data.xlsx or yahoo_data.xlsx.")




def minmax_scale(train_values, value_x):
   train_min = train_values.min(axis=0)
   train_max = train_values.max(axis=0)
   m_range = train_max - train_min
   m_range[m_range == 0] = 1.0
   scaled_all = (value_x - train_min) / m_range
   return scaled_all, train_min, train_max




def min_max_scale(scaled_target, target_min, target_max):
   return scaled_target * (target_max - target_min) + target_min




def create_sequences(data, target_index, sequence_length):
   X, y = [], []
   for i in range(len(data) - sequence_length):
       X.append(data[i:i + sequence_length])
       y.append(data[i + sequence_length, target_index])
   return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)




def calculate_metrics(actual, predicted):
   mse = np.mean((actual - predicted) ** 2)
   rmse = math.sqrt(mse)
   mae = np.mean(np.abs(actual - predicted))
   mape = np.mean(np.abs((actual - predicted) / actual)) * 100
   return mse, rmse, mae, mape




class LSTM_reg(nn.Module):
   def __init__(self, input_size, hidden_size):
       super().__init__()
       self.hidden_size = hidden_size
       self.x_to_gates = nn.Linear(input_size, 4 * hidden_size)
       self.h_to_gates = nn.Linear(hidden_size, 4 * hidden_size)
       self.fc = nn.Linear(hidden_size, 1)


   def forward(self, x):
       batch_size, seq_len, _ = x.shape


       h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
       c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)


       for t in range(seq_len):
           x_t = x[:, t, :]


           gates = self.x_to_gates(x_t) + self.h_to_gates(h_t)
           f_t, i_t, g_t, o_t = torch.chunk(gates, chunks=4, dim=1)


           f_t = torch.sigmoid(f_t)
           i_t = torch.sigmoid(i_t)
           g_t = torch.tanh(g_t)
           o_t = torch.sigmoid(o_t)


           c_t = f_t * c_t + i_t * g_t
           h_t = o_t * torch.tanh(c_t)


       return self.fc(h_t)




data_file = find_data()
df = pd.read_excel(data_file)


df.columns = [str(c).strip() for c in df.columns]
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


feature_columns = ["Open", "High", "Low", "Close*", "Adj Close**", "Volume"]
target_index = feature_columns.index(target_c)


data_values = df[feature_columns].values.astype(np.float32)


raw_train_size = int(len(data_values) * t_ratio)
train_raw = data_values[:raw_train_size]


scaled_values, train_min, train_max = minmax_scale(train_raw, data_values)


X, y = create_sequences(scaled_values, target_index, sequence_length)


train_size = int(len(X) * t_ratio)


X_train = torch.tensor(X[:train_size])
y_train = torch.tensor(y[:train_size])
X_test = torch.tensor(X[train_size:])
y_test = torch.tensor(y[train_size:])


print(f"Data file: {data_file}")
print(f"Dataset rows: {len(df)}")
print(f"Sequence length: {sequence_length}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


model = LSTM_reg(input_size=len(feature_columns), hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


loss_history = []


for epoch in range(1, EPOCHS + 1):
   model.train()
   optimizer.zero_grad()


   train_predictions = model(X_train)
   loss = criterion(train_predictions, y_train)


   loss.backward()
   optimizer.step()


   loss_history.append(loss.item())


   if epoch == 1 or epoch % 10 == 0:
       print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.6f}")


model.eval()
with torch.no_grad():
   test_predictions_scaled = model(X_test).cpu().numpy().flatten()
   y_test_scaled = y_test.cpu().numpy().flatten()


target_min = train_min[target_index]
target_max = train_max[target_index]


test_predictions = min_max_scale(test_predictions_scaled, target_min, target_max)
actual_values = min_max_scale(y_test_scaled, target_min, target_max)


mse, rmse, mae, mape = calculate_metrics(actual_values, test_predictions)


print("\nLSTM Test Results")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.4f}%")


plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig("lstm_training_loss.png", dpi=300)
plt.close()


test_dates = df["Date"].iloc[sequence_length + train_size:].reset_index(drop=True)


plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_values, label="Actual Close Price")
plt.plot(test_dates, test_predictions, label="Predicted Close Price")
plt.title("LSTM Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("lstm_prediction_result.png", dpi=300)
plt.close()


log_text = f"""Experiment 2
Model: Manual LSTM
Target column: {target_c}
Sequence length: {sequence_length}
Hidden size: {HIDDEN_SIZE}
Learning rate: {LEARNING_RATE}
Epochs: {EPOCHS}
Loss function: MSE
Optimizer: Adam
Train/Test Split: {int(t_ratio * 100)}/{int((1 - t_ratio) * 100)}
Dataset size: {len(df)}
Test MSE: {mse:.4f}
Test RMSE: {rmse:.4f}
Test MAE: {mae:.4f}
Test MAPE: {mape:.4f}%
"""


with open("lstm_experiment_log.txt", "w", encoding="utf-8") as f:
   f.write(log_text)


print("\nLSTM experiment log saved.")
print("Plots saved.")
