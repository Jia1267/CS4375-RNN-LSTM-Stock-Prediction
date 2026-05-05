import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

## Set random seed for reproducibility
np.random.seed(42)

## RNN implementation for stock price prediction
def read_data():
    file_name = "stock_data.xlsx"
    df = pd.read_excel(file_name)

    print("Dataset loaded.")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    return df

## Clean and prepare the data for modeling
def clean_data(df):
    df = df.copy()

    ## Ensure 'Date' column is datetime and sort by date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    cols = ["Open", "High", "Low", "Close*", "Adj Close**", "Volume"]

    df = df[cols]
    df = df.dropna()

    print("\nFeatures used:", cols)
    print("Target column: Close*")
    print("Cleaned shape:", df.shape)

    return df, cols

## Normalize the data using training set statistics
def normalize(train_data, test_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std == 0] = 1

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    return train_data, test_data, mean, std

## Create sequences of data for RNN input
def make_sequences(data, target_col, seq_len):
    X = []
    y = []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, target_col])

    return np.array(X), np.array(y).reshape(-1, 1)

## Simple RNN implementation from scratch
class SimpleRNN:
    def __init__(self, input_size, hidden_size, lr):
        self.hidden_size = hidden_size
        self.lr = lr

        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, 1) * 0.01

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, 1))

    ## Forward pass through the RNN
    def forward(self, x):
        h_list = []
        h_prev = np.zeros((1, self.hidden_size))

        for t in range(x.shape[0]):
            x_t = x[t].reshape(1, -1)

            h = np.tanh(
                np.dot(x_t, self.Wxh)
                + np.dot(h_prev, self.Whh)
                + self.bh
            )

            h_list.append(h)
            h_prev = h

        out = np.dot(h_list[-1], self.Why) + self.by

        return out, h_list
    
    ## Backward pass to compute gradients and update weights
    def backward(self, x, y, pred, h_list):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dy = 2 * (pred - y)

        dWhy += np.dot(h_list[-1].T, dy)
        dby += dy

        dh = np.dot(dy, self.Why.T)

        for t in reversed(range(x.shape[0])):
            h = h_list[t]

            if t == 0:
                h_prev = np.zeros((1, self.hidden_size))
            else:
                h_prev = h_list[t - 1]

            x_t = x[t].reshape(1, -1)

            dtanh = dh * (1 - h ** 2)

            dbh += dtanh
            dWxh += np.dot(x_t.T, dtanh)
            dWhh += np.dot(h_prev.T, dtanh)

            dh = np.dot(dtanh, self.Whh.T)

        for g in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(g, -5, 5, out=g)

        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    ## Train the RNN on the provided data
    def train(self, X, y, epochs):
        losses = []

        for e in range(epochs):
            total_loss = 0

            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i].reshape(1, -1)

                pred, h_list = self.forward(x_i)
                loss = np.mean((pred - y_i) ** 2)

                total_loss += loss
                self.backward(x_i, y_i, pred, h_list)

            avg_loss = total_loss / len(X)
            losses.append(avg_loss)

            print(f"Epoch {e + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses
    
    ## Predict using the trained RNN model
    def predict(self, X):
        result = []

        for i in range(len(X)):
            pred, _ = self.forward(X[i])
            result.append(pred[0, 0])

        return np.array(result)


def main():
    df = read_data()
    df, features = clean_data(df)

    data = df.values.astype(float)
    target_index = features.index("Close*")

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_norm, test_norm, mean, std = normalize(train_data, test_data)

    ## Create sequences for RNN input
    seq_len = 20
    X_train, y_train = make_sequences(train_norm, target_index, seq_len)
    X_test, y_test = make_sequences(test_norm, target_index, seq_len)

    print("\nSequence data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    ## Train the RNN model
    model = SimpleRNN(
        input_size=X_train.shape[2],
        hidden_size=32,
        lr=0.01
    )

    ## Train the model and evaluate on the test set
    losses = model.train(X_train, y_train, epochs=50)

    pred_norm = model.predict(X_test)
    actual_norm = y_test.flatten()

    close_mean = mean[target_index]
    close_std = std[target_index]

    pred = pred_norm * close_std + close_mean
    actual = actual_norm * close_std + close_mean

    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)

    ## Print final test results and visualize predictions
    print("\nFinal Test Results")
    print("============================")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    ## Plot training loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("RNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

    ## Plot actual vs predicted close prices
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual Close Price")
    plt.plot(pred, label="Predicted Close Price")
    plt.title("RNN Stock Price Prediction")
    plt.xlabel("Test Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("rnn_prediction_result.png")
    plt.show()


if __name__ == "__main__":
    main()