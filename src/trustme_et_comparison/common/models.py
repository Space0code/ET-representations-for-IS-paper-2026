"""Model factory and PyTorch MLP classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .types import ExperimentConfig, MLPConfig
from .utils import set_global_seed


class MLPNet(nn.Module):
    """Simple feed-forward network with LayerNorm + GELU blocks."""

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, use_layernorm: bool) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(inputs)


@dataclass
class TorchMLPClassifier:
    """Sklearn-like classifier wrapper around PyTorch MLP."""

    config: MLPConfig
    input_dim: int
    num_classes: int

    def __post_init__(self) -> None:
        set_global_seed(self.config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPNet(
            input_dim=self.input_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=self.num_classes,
            use_layernorm=self.config.use_layernorm,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def _build_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "TorchMLPClassifier":
        """Train MLP with early stopping on validation loss."""
        train_loader = self._build_loader(X_train, y_train, shuffle=True)
        val_loader = self._build_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for _ in range(self.config.max_epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(self.device)
                    val_y = val_y.to(self.device)
                    val_logits = self.model(val_X)
                    val_loss = self.loss_fn(val_logits, val_y)
                    val_losses.append(float(val_loss.detach().cpu().item()))

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        self.model.eval()
        loader = DataLoader(
            torch.as_tensor(X, dtype=torch.float32),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                batch_probs = torch.softmax(logits, dim=1)
                probs.append(batch_probs.detach().cpu().numpy())

        return np.vstack(probs).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(np.int64)


def create_model(
    model_name: str,
    cfg: ExperimentConfig,
    input_dim: int,
    num_classes: int,
):
    """Instantiate a classifier by name."""
    if model_name == "majority":
        return DummyClassifier(**cfg.models.majority)

    if model_name == "logistic_regression":
        params = dict(cfg.models.logistic_regression)
        # sklearn>=1.8 ignores n_jobs for lbfgs and warns; drop if provided.
        params.pop("n_jobs", None)
        params.setdefault("max_iter", 2000)
        return LogisticRegression(**params)

    if model_name == "lgbm":
        return LGBMClassifier(**cfg.models.lgbm)

    if model_name == "svm_rbf":
        return SVC(**cfg.models.svm_rbf)

    if model_name == "random_forest":
        return RandomForestClassifier(**cfg.models.random_forest)

    if model_name == "mlp":
        return TorchMLPClassifier(cfg.models.mlp, input_dim=input_dim, num_classes=num_classes)

    raise ValueError(f"Unknown model name: {model_name}")
