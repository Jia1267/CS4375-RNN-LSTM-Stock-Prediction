from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================================================
# 1. Load Kaggle CSV data
# =========================================================

def load_stock_data(filename="stock_data.xlsx"):
    """
    Load stock data from a Kaggle Excel file.
    """

    df = pd.read_excel(filename)

    print("Dataset loaded successfully.")
    print("Columns in dataset:")
    print(df.columns.tolist())
    print("Dataset shape:", df.shape)

    return df


# =========================================================
# 2. Preprocess data
# =========================================================

def preprocess_data(df):
    """
    Clean the Kaggle Yahoo Finance stock dataset.
    This dataset uses Close* and Adj Close** as column names.
    """

    df = df.copy()

    # Remove unnamed index columns if they exist
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Convert Date column and sort by date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # Features in your Kaggle file
    possible_features = [
        "Open", "High", "Low", "Close*", "Adj Close**", "Volume",
        "Close", "Adj Close",
        "open", "high", "low", "close", "adj close", "volume"
    ]

    features = [col for col in possible_features if col in df.columns]

    if len(features) == 0:
        raise ValueError("No valid stock price columns were found.")

    # Your dataset uses Close*
    if "Close*" in df.columns:
        target_column = "Close*"
    elif "Close" in df.columns:
        target_column = "Close"
    elif "close" in df.columns:
        target_column = "close"
    else:
        raise ValueError("The dataset must contain a Close, Close*, or close column.")

    df = df[features]
    df = df.dropna()

    print("\nFeatures used:")
    print(features)
    print("Target column:", target_column)
    print("Cleaned dataset shape:", df.shape)

    return df, features, target_column


def train_test_split_time_series(data, train_ratio=0.8):
    """
    For time series prediction, do not shuffle data.
    Earlier data is training data, later data is test data.
    """

    train_size = int(len(data) * train_ratio)

    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data


def normalize_train_test(train_data, test_data):
    """
    Normalize data using only training mean and std.
    This avoids data leakage.
    """

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    std[std == 0] = 1

    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std

    return train_norm, test_norm, mean, std


def create_sequences(data, target_index, sequence_length=20):
    """
    Create sequences for RNN.

    Example:
    use previous 20 days to predict the next day's Close price.
    """

    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_index])

    return np.array(X), np.array(y).reshape(-1, 1)


# =========================================================
# 3. Simple RNN implemented from scratch
# =========================================================

class SimpleRNN:
    def __init__(self, input_size, hidden_size=32, output_size=1, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    def forward(self, x):
        """
        Forward pass for one sequence.

        x shape:
        sequence_length x input_size
        """

        h_states = []
        h_prev = np.zeros((1, self.hidden_size))

        for t in range(x.shape[0]):
            x_t = x[t].reshape(1, -1)

            h_t = np.tanh(
                np.dot(x_t, self.Wxh)
                + np.dot(h_prev, self.Whh)
                + self.bh
            )

            h_states.append(h_t)
            h_prev = h_t

        y_pred = np.dot(h_states[-1], self.Why) + self.by

        return y_pred, h_states

    def backward(self, x, y_true, y_pred, h_states):
        """
        Backpropagation Through Time.
        """

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dy = 2 * (y_pred - y_true)

        dWhy += np.dot(h_states[-1].T, dy)
        dby += dy

        dh_next = np.dot(dy, self.Why.T)

        for t in reversed(range(x.shape[0])):
            h_t = h_states[t]

            if t == 0:
                h_prev = np.zeros((1, self.hidden_size))
            else:
                h_prev = h_states[t - 1]

            x_t = x[t].reshape(1, -1)

            dtanh = dh_next * (1 - h_t ** 2)

            dbh += dtanh
            dWxh += np.dot(x_t.T, dtanh)
            dWhh += np.dot(h_prev.T, dtanh)

            dh_next = np.dot(dtanh, self.Whh.T)

        # Prevent exploding gradient
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, X_train, y_train, epochs=50):
        losses = []

        for epoch in range(epochs):
            total_loss = 0

            for i in range(len(X_train)):
                x = X_train[i]
                y_true = y_train[i].reshape(1, -1)

                y_pred, h_states = self.forward(x)

                loss = np.mean((y_pred - y_true) ** 2)
                total_loss += loss

                self.backward(x, y_true, y_pred, h_states)

            avg_loss = total_loss / len(X_train)
            losses.append(avg_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.6f}")

        return losses

    def predict(self, X):
        predictions = []

        for i in range(len(X)):
            y_pred, _ = self.forward(X[i])
            predictions.append(y_pred[0, 0])

        return np.array(predictions)


# =========================================================
# 4. Evaluation
# =========================================================

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return mse, rmse, mae


def save_experiment_log(
    filename,
    sequence_length,
    hidden_size,
    learning_rate,
    epochs,
    train_size,
    test_size,
    mse,
    rmse,
    mae
):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        f.write("=" * 60 + "\n")
        f.write("Model: Simple RNN from scratch using NumPy\n")
        f.write("Dataset: Kaggle Yahoo Finance Dataset 2018-2023\n")
        f.write(f"Sequence Length: {sequence_length}\n")
        f.write(f"Hidden Size: {hidden_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write("----------------------------\n")
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Test RMSE: {rmse:.6f}\n")
        f.write(f"Test MAE: {mae:.6f}\n")


# =========================================================
# 5. Main
# =========================================================

def main():
    filename = "stock_data.xlsx"

    sequence_length = 20
    hidden_size = 32
    learning_rate = 0.001
    epochs = 50

    df = load_stock_data(filename)

    df, features, target_column = preprocess_data(df)

    data = df.values.astype(float)
    target_index = features.index(target_column)

    train_data, test_data = train_test_split_time_series(data, train_ratio=0.8)

    train_norm, test_norm, mean, std = normalize_train_test(train_data, test_data)

    X_train, y_train = create_sequences(
        train_norm,
        target_index=target_index,
        sequence_length=sequence_length
    )

    X_test, y_test = create_sequences(
        test_norm,
        target_index=target_index,
        sequence_length=sequence_length
    )

    print("\nSequence data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    model = SimpleRNN(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        output_size=1,
        learning_rate=learning_rate
    )

    losses = model.train(X_train, y_train, epochs=epochs)

    y_pred_norm = model.predict(X_test)
    y_test_norm = y_test.flatten()

    close_mean = mean[target_index]
    close_std = std[target_index]

    y_pred = y_pred_norm * close_std + close_mean
    y_true = y_test_norm * close_std + close_mean

    mse, rmse, mae = evaluate_model(y_true, y_pred)

    print("\nFinal Test Results")
    print("============================")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    save_experiment_log(
        filename="experiment_log.txt",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
        train_size=len(X_train),
        test_size=len(X_test),
        mse=mse,
        rmse=rmse,
        mae=mae
    )

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("RNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual Close Price")
    plt.plot(y_pred, label="Predicted Close Price")
    plt.title("RNN Stock Price Prediction")
    plt.xlabel("Test Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("rnn_prediction_result.png")
    plt.show()


if __name__ == "__main__":
    main()