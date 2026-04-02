import numpy as np
import pandas as pd

from utils import prepare_features
from utils import minmax_fit, minmax_transform, minmax_inverse_transform
from utils import mae, mse, rmse, r2_score


np.random.seed(42)


class MLPRegressorScratch:
    def __init__(
        self,
        input_size,
        hidden_size=16,
        output_size=1,
        learning_rate=0.01,
        epochs=8000,
        momentum=0.9,
        tol=1e-10,
        patience=800,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.tol = tol
        self.patience = patience

        # Xavier-like initialization
        limit_w1 = np.sqrt(6 / (input_size + hidden_size))
        limit_w2 = np.sqrt(6 / (hidden_size + output_size))

        self.W1 = np.random.uniform(-limit_w1, limit_w1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.uniform(-limit_w2, limit_w2, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        # Momentum terms
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        self.loss_history = []
        self.final_epoch = 0
        self.final_loss = None
        self.training_end_reason = ""

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, a):
        return 1 - np.square(a)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.z2  # linear output for regression
        return self.y_hat

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred):
        n = X.shape[0]

        dL_dyhat = (2 / n) * (y_pred - y_true)

        dW2 = np.dot(self.a1.T, dL_dyhat)
        db2 = np.sum(dL_dyhat, axis=0, keepdims=True)

        d_hidden = np.dot(dL_dyhat, self.W2.T) * self.tanh_derivative(self.a1)
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.vW1 = self.momentum * self.vW1 - self.learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1
        self.vW2 = self.momentum * self.vW2 - self.learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def fit(self, X, y, verbose=True):
        best_loss = float("inf")
        no_improvement_count = 0

        for epoch in range(1, self.epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            self.update_parameters(dW1, db1, dW2, db2)

            improvement = best_loss - loss
            if improvement > self.tol:
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if verbose and (epoch == 1 or epoch % 500 == 0):
                print("Epoch {}/{} - Loss: {:.8f}".format(epoch, self.epochs, loss))

            if no_improvement_count >= self.patience:
                print("Early stopping at epoch {}. Best loss: {:.8f}".format(epoch, best_loss))
                self.training_end_reason = "Early stopping triggered"
                break

        if self.training_end_reason == "":
            self.training_end_reason = "Maximum epoch limit reached"

        self.final_epoch = epoch
        self.final_loss = self.loss_history[-1]

    def predict(self, X):
        return self.forward(X)


def write_report(report_path, model, feature_names, target_col, train_metrics, test_metrics):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PART A - MLP REGRESSION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 60 + "\n")
        f.write("Input features after preprocessing: {}\n".format(feature_names))
        f.write("Target variable: {}\n".format(target_col))
        f.write("Normalization: Min-Max normalization\n")
        f.write("Categorical preprocessing: One-hot encoding for Neighborhood\n")
        f.write("Test normalization rule: Test set normalized using TRAIN set min-max values\n\n")

        f.write("2. NETWORK ARCHITECTURE\n")
        f.write("-" * 60 + "\n")
        f.write("Input layer neurons: {}\n".format(model.input_size))
        f.write("Number of hidden layers: 1\n")
        f.write("Hidden layer neurons: {}\n".format(model.hidden_size))
        f.write("Output layer neurons: {}\n".format(model.output_size))
        f.write("Hidden layer activation: tanh\n")
        f.write("Output layer activation: linear\n\n")

        f.write("3. TRAINING PARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write("Initial learning rate: {}\n".format(model.learning_rate))
        f.write("Momentum: {}\n".format(model.momentum))
        f.write("Weight update method: Gradient Descent with Momentum\n")
        f.write("Maximum epochs: {}\n".format(model.epochs))
        f.write("Actual training epochs: {}\n".format(model.final_epoch))
        f.write(
            "Stopping criterion: Early stopping with tolerance={}, patience={}\n".format(
                model.tol, model.patience
            )
        )
        f.write("Training end reason: {}\n".format(model.training_end_reason))
        f.write("Final training loss (normalized MSE): {:.8f}\n\n".format(model.final_loss))

        f.write("4. TRAIN RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write("MAE: {:.6f}\n".format(train_metrics["MAE"]))
        f.write("MSE: {:.6f}\n".format(train_metrics["MSE"]))
        f.write("RMSE: {:.6f}\n".format(train_metrics["RMSE"]))
        f.write("R2 (coefficient of determination): {:.6f}\n\n".format(train_metrics["R2"]))

        f.write("5. TEST RESULTS (MACRO AVERAGE)\n")
        f.write("-" * 60 + "\n")
        f.write("MAE: {:.6f}\n".format(test_metrics["MAE"]))
        f.write("MSE: {:.6f}\n".format(test_metrics["MSE"]))
        f.write("RMSE: {:.6f}\n".format(test_metrics["RMSE"]))
        f.write("R2 (coefficient of determination): {:.6f}\n\n".format(test_metrics["R2"]))

        f.write("6. FINAL WEIGHTS AND BIASES\n")
        f.write("-" * 60 + "\n")
        f.write("W1 shape: {}\n".format(model.W1.shape))
        f.write("W1:\n")
        f.write(np.array2string(model.W1, precision=6, suppress_small=False))

        f.write("\n\nb1 shape: {}\n".format(model.b1.shape))
        f.write("b1:\n")
        f.write(np.array2string(model.b1, precision=6, suppress_small=False))

        f.write("\n\nW2 shape: {}\n".format(model.W2.shape))
        f.write("W2:\n")
        f.write(np.array2string(model.W2, precision=6, suppress_small=False))

        f.write("\n\nb2 shape: {}\n".format(model.b2.shape))
        f.write("b2:\n")
        f.write(np.array2string(model.b2, precision=6, suppress_small=False))
        f.write("\n\n")

        f.write("7. COMMENTS\n")
        f.write("-" * 60 + "\n")
        f.write("The MLP model was implemented from scratch using only NumPy and Pandas.\n")
        f.write("All input variables and the target variable were normalized with min-max normalization.\n")
        f.write("Since the Neighborhood attribute is categorical, one-hot encoding was applied before normalization.\n")
        f.write("The model was trained with back-propagation and momentum-based weight updates.\n")
        f.write("Performance was evaluated on both training and test sets using MAE, MSE, RMSE, and R2.\n")
        f.write(
            "The relatively low R2 values indicate that the relationship between the input features "
            "and house prices could only be captured partially by the selected network architecture "
            "and training configuration.\n"
        )


def main():
    train_path = "data/midtermProject-part1-train.xlsx"
    test_path = "data/midtermProject-part1-test.xlsx"
    report_path = "report.txt"

    print("Loading datasets.")
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)

    feature_cols = list(train_df.columns[:3])
    target_col = train_df.columns[3]

    X_train, X_test, feature_names = prepare_features(train_df, test_df, feature_cols)

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").values.reshape(-1, 1).astype(float)
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").values.reshape(-1, 1).astype(float)

    X_min, X_max = minmax_fit(X_train)
    y_min, y_max = minmax_fit(y_train)

    X_train_norm = minmax_transform(X_train, X_min, X_max)
    X_test_norm = minmax_transform(X_test, X_min, X_max)
    y_train_norm = minmax_transform(y_train, y_min, y_max)

    print("\nNormalization completed.")
    print("X_train_norm shape:", X_train_norm.shape)
    print("y_train_norm shape:", y_train_norm.shape)

    model = MLPRegressorScratch(
        input_size=X_train_norm.shape[1],
        hidden_size=16,
        output_size=1,
        learning_rate=0.01,
        epochs=8000,
        momentum=0.9,
        tol=1e-10,
        patience=800,
    )

    print("\nTraining model.")
    model.fit(X_train_norm, y_train_norm, verbose=True)

    train_pred_norm = model.predict(X_train_norm)
    test_pred_norm = model.predict(X_test_norm)

    train_pred = minmax_inverse_transform(train_pred_norm, y_min, y_max)
    test_pred = minmax_inverse_transform(test_pred_norm, y_min, y_max)

    train_metrics = {
        "MAE": mae(y_train, train_pred),
        "MSE": mse(y_train, train_pred),
        "RMSE": rmse(y_train, train_pred),
        "R2": r2_score(y_train, train_pred),
    }

    test_metrics = {
        "MAE": mae(y_test, test_pred),
        "MSE": mse(y_test, test_pred),
        "RMSE": rmse(y_test, test_pred),
        "R2": r2_score(y_test, test_pred),
    }

    print("\nTrain Results")
    print("MAE :", train_metrics["MAE"])
    print("MSE :", train_metrics["MSE"])
    print("RMSE:", train_metrics["RMSE"])
    print("R2  :", train_metrics["R2"])

    print("\nTest Results")
    print("MAE :", test_metrics["MAE"])
    print("MSE :", test_metrics["MSE"])
    print("RMSE:", test_metrics["RMSE"])
    print("R2  :", test_metrics["R2"])

    write_report(report_path, model, feature_names, target_col, train_metrics, test_metrics)
    print("\nReport saved to:", report_path)


if __name__ == "__main__":
    main()