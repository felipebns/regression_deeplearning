from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class mlp:
    def __init__(self, n_features:int, n_hidden_layers: int, n_neurons_per_layer: list, 
                activation: str, loss: str, optimizer: str, epochs: int, eta: float) -> None:
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.eta = eta

        self.weights = []
        self.biases = []
        self.layer_dims = [n_features] + n_neurons_per_layer

        for i in range(len(self.layer_dims) - 1):
            w = np.random.randn(self.layer_dims[i+1], self.layer_dims[i]) * 0.1
            b = np.zeros((self.layer_dims[i+1],))
            self.weights.append(w)
            self.biases.append(b)

        # histórico de treino
        self.metric_name = "Accuracy" if loss == "cross_entropy" else "RMSE"
        self.history = {"loss": [], "metric": []}

        print("\n=== Inicialização de Pesos e Biases ===")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Camada {i+1}:")
            print(f"W{i+1} shape {w.shape}:\n{w}")
            print(f"b{i+1} shape {b.shape}:\n{b}\n")

    def train(self, X, y, threshold: float = 5e-3, window: int = 10) -> None:
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(y)):
                y_pred, cache = self.forward_pass(X[i])
                loss = self.loss_calculation(y[i], y_pred)
                total_loss += loss
                grads_w, grads_b = self.backpropagation(y[i], y_pred, cache)
                self.update_parameters(grads_w, grads_b)

            avg_loss = total_loss / len(y)
            loss_history.append(avg_loss)

            preds_train = self.test(X)
            if self.loss == "cross_entropy":
                train_metric = self.calculate_accuracy(y, preds_train)
            else:
                train_metric = self.calculate_rmse(y, preds_train)

            # armazenar no histórico
            self.history["loss"].append(avg_loss)
            self.history["metric"].append(train_metric)

            if epoch % 10 == 0:
                if self.loss == "cross_entropy":
                    metric_str = f"{train_metric*100:.2f}%" if not np.isnan(train_metric) else "nan"
                else:
                    metric_str = f"{train_metric:.6f}"
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Train {self.metric_name}: {metric_str}")

            # critério de parada: média móvel dos últimos "window" epochs
            if epoch >= window:
                moving_avg_prev = np.mean(loss_history[-2*window:-window])
                moving_avg_curr = np.mean(loss_history[-window:])
                if abs(moving_avg_prev - moving_avg_curr) < threshold:
                    print(f"Treinamento encerrado no epoch {epoch} (convergência detectada).")
                    break

    def test(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(len(X)):
            y_pred, _ = self.forward_pass(X[i])
            if self.loss == "cross_entropy":
                preds.append(np.argmax(y_pred))
            else:
                val = np.array(y_pred).ravel()
                preds.append(val[0] if val.size > 0 else val)
        return np.array(preds)


    def evaluate(self, X: np.ndarray, y: np.ndarray, plot_confusion: bool, plot_roc: bool, preds: np.ndarray):
        if self.loss == "cross_entropy":
            acc = self.calculate_accuracy(y, preds)
            print(f"Accuracy: {acc*100:.2f}%")
            report = {"accuracy": acc}
            self.last_classification_report = report
        else:
            y_true = np.asarray(y).ravel()
            preds = np.asarray(preds).ravel()
            metrics = self._compute_regression_metrics(y_true, preds)

            baseline_preds = np.full_like(y_true, y_true.mean())
            baseline_metrics = self._compute_regression_metrics(y_true, baseline_preds)

            self._print_regression_summary(metrics, baseline_metrics)

            report = {
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "residuals": y_true - preds,
                "y_true": y_true,
                "y_pred": preds,
            }
            self.last_regression_report = report

        print("\n=== Pesos e Biases do Modelo ===")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"\nCamada {i+1}:")
            print(f"  Pesos W{i+1} (shape {w.shape}):")
            print(w)
            print(f"  Biases b{i+1} (shape {b.shape}):")
            print(b)

        binary = (len(np.unique(y)) == 2)

        if plot_confusion and self.loss == "cross_entropy":
            self.plot_confusion_matrix(y, preds)
        elif plot_confusion:
            print("Confusion matrix não é aplicável para regressão.")

        if plot_roc and binary and self.loss == "cross_entropy":
            self.plot_roc_curve(X, y)
        elif plot_roc and self.loss != "cross_entropy":
            print("ROC curve não é aplicável para regressão.")

        # sempre plota histórico se existir
        if self.history and len(self.history.get("loss", [])) > 0:
            self.plot_history()

        return report

    # -------- Funções auxiliares de plot --------
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6,5))
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title("Confusion Matrix (normalized by row)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def plot_roc_curve(self, X, y_true):
        y_scores = []
        for i in range(len(X)):
            y_pred, _ = self.forward_pass(X[i])
            if self.loss == "cross_entropy":
                y_scores.append(y_pred[1])
            else:
                val = np.array(y_pred).ravel()
                y_scores.append(val[0] if val.size > 0 else val)
        y_scores = np.array(y_scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()

    def plot_history(self):
        """Plota loss e accuracy armazenados em self.history."""
        loss = self.history.get("loss", [])
        metric = self.history.get("metric", [])
        epochs = np.arange(1, len(loss)+1)

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        # loss
        axes[0].plot(epochs, loss, marker="o")
        axes[0].set_title("Loss por Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        # accuracy
        axes[1].plot(epochs, metric, marker="o")
        axes[1].set_title(f"{self.metric_name} por Epoch (treino)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(self.metric_name)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def _compute_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MAPE (%)": mape,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }

    def _print_regression_summary(self, model_metrics: dict, baseline_metrics: dict) -> None:
        header = f"{'Metric':<12}{'Model':>14}{'Baseline':>14}"
        print("\nRegression metrics (model vs. mean baseline):")
        print(header)
        print("-" * len(header))
        for key in ["MAE", "MAPE (%)", "MSE", "RMSE", "R2"]:
            model_val = model_metrics.get(key, float("nan"))
            baseline_val = baseline_metrics.get(key, float("nan"))
            print(f"{key:<12}{model_val:>14.6f}{baseline_val:>14.6f}")

    def forward_pass(self, x: np.ndarray) -> tuple:
        a = x
        cache = {"z": [], "a": [a]}
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            if i == len(self.weights) - 1 and self.loss == "cross_entropy":
                a = self.softmax(z)
            else:
                a = self.activation_function(z)
            cache["z"].append(z)
            cache["a"].append(a)
        return a, cache

    def backpropagation(self, y_true: np.ndarray, y_pred: np.ndarray, cache: dict) -> tuple:
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Última camada
        if self.loss == "cross_entropy":
            delta = self.derive_cross_entropy(y_true, y_pred)
        elif self.loss == "mse":
            dloss_dy_pred = self.derive_mse(y_true, y_pred)
            if self.activation == "sigmoid":
                delta = dloss_dy_pred * self.derive_sigmoid(cache["z"][-1])
            elif self.activation == "tanh":
                delta = dloss_dy_pred * self.derive_tanh(cache["z"][-1])
            elif self.activation == "relu":
                delta = dloss_dy_pred * self.derive_relu(cache["z"][-1])
        else:
            raise ValueError("Loss não suportada")

        grads_w[-1] = np.outer(delta, cache["a"][-2])
        grads_b[-1] = delta

        # Camadas ocultas
        # não faz sentido atualizar delta em "a", já que só faz sentido atualizar os pesos, que estão na pré ativação "z"
        for l in reversed(range(len(self.weights)-1)):
            delta = np.dot(self.weights[l+1].T, delta)
            if self.activation == "sigmoid":
                delta *= self.derive_sigmoid(cache["z"][l])
            elif self.activation == "tanh":
                delta *= self.derive_tanh(cache["z"][l])
            elif self.activation == "relu":
                delta *= self.derive_relu(cache["z"][l])
            grads_w[l] = np.outer(delta, cache["a"][l]) # o primeiro x esta dentro do cache "a"
            grads_b[l] = delta

        return grads_w, grads_b
        
    def update_parameters(self, grads_w, grads_b):
        if self.optimizer == "gd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.eta * grads_w[i]
                self.biases[i]  -= self.eta * grads_b[i]
        else:
            raise ValueError(f"Optimizer {self.optimizer} não suportado")

    def loss_calculation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.loss == 'mse':
            return self.mse(y_true, y_pred)
        elif self.loss == 'cross_entropy':
            return self.cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Função de loss {self.loss} não suportada")

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'relu':
            return self.relu(z)
        else:
            raise ValueError(f"Função de ativação {self.activation} não suportada")
        
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)

    def derive_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2*(y_true - y_pred)

    def cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        num_classes = len(y_pred)
        y_true_onehot = np.eye(num_classes)[y_true]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true_onehot * np.log(y_pred))

    def derive_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        num_classes = len(y_pred)
        y_true_onehot = np.eye(num_classes)[y_true]
        return y_pred - y_true_onehot

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def derive_sigmoid(self, z: np.ndarray) -> np.ndarray:
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derive_tanh(self, z: np.ndarray) -> np.ndarray:
        return 1 - (np.tanh(z))**2

    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derive_relu(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
