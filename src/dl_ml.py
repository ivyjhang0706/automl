import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVR
from pytorch_tabnet.tab_model import TabNetRegressor as _TabNetRegressor

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class _ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        # 在 Tabular 特徵上，LayerNorm 比 BatchNorm1d 更穩定且不依賴 Batch Size
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class _ResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_blocks, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x)


class ResNetMLPRegressor(RegressorMixin, BaseEstimator):
    """ResNetMLPClassifier 的回歸版：同樣的 Residual Block 架構，輸出改成單一連續值 + MSELoss。
    標籤在內部做標準化（NN 對輸出量級敏感），predict 時再還原回原始尺度。"""

    def __init__(self, hidden_dim=64, n_blocks=2, dropout=0.2, lr=1e-3,
                 epochs=30, batch_size=32, weight_decay=1e-5, device=_DEVICE):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.x_scaler_ = StandardScaler().fit(X)
        X = self.x_scaler_.transform(X)

        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        self.model_ = _ResNetMLP(X.shape[1], self.hidden_dim, 1, self.n_blocks, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model_(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X = self.x_scaler_.transform(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            pred_norm = self.model_(X_t).squeeze(-1).cpu().numpy()
        return pred_norm * self.y_std_ + self.y_mean_


class TabNetRegressorWrapper(RegressorMixin, BaseEstimator):
    """把 pytorch-tabnet 的 TabNetRegressor 包成標準 sklearn 回歸介面，方便丟進 cross_val_score。
    X/y 一律轉成 numpy array；TabNetRegressor 額外要求 y 是 2D shape (n_samples, 1)。"""

    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, lr=2e-2,
                 max_epochs=50, batch_size=256, device=_DEVICE):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.model_ = _TabNetRegressor(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma,
            optimizer_params=dict(lr=self.lr),
            device_name=self.device, verbose=0, seed=42,
        )
        # 沒給 eval_set 就沒有 early stopping 依據，patience=max_epochs 讓它跑滿。
        self.model_.fit(X, y, max_epochs=self.max_epochs, patience=self.max_epochs,
                         batch_size=self.batch_size)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float32)).reshape(-1)
