"""
Deep Learning Models for Time Series Forecasting.

Enterprise-grade implementations:
- LSTM/GRU networks
- Transformer models
- Attention mechanisms
- Multi-step forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from source.utils.logger import setup_logger

logger = setup_logger(name="deep_learning", log_file="../logs/deep_learning.log")


@dataclass
class ModelConfig:
    """Deep learning model configuration."""
    input_size: int = 10
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    sequence_length: int = 24


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialize dataset."""
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.

    Features:
    - Bidirectional LSTM
    - Dropout regularization
    - Multi-layer architecture
    - Attention mechanism (optional)
    """

    def __init__(self, config: ModelConfig):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last output
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            out = hidden[-1]

        out = self.dropout(out)
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    """
    GRU model for time series forecasting.

    Similar to LSTM but with fewer parameters and faster training.
    """

    def __init__(self, config: ModelConfig):
        """Initialize GRU model."""
        super(GRUModel, self).__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        gru_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gru_out, hidden = self.gru(x)

        if self.config.bidirectional:
            out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            out = hidden[-1]

        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting.

    Features:
    - Multi-head self-attention
    - Positional encoding
    - Feed-forward networks
    """

    def __init__(self, config: ModelConfig, num_heads: int = 8):
        """Initialize Transformer model."""
        super(TransformerModel, self).__init__()
        self.config = config

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            config.sequence_length,
            config.input_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_size,
            nhead=num_heads,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Transformer encoding
        transformer_out = self.transformer_encoder(x)

        # Use mean pooling
        out = transformer_out.mean(dim=1)

        # Output layer
        out = self.fc(out)

        return out


class DeepLearningTrainer:
    """
    Trainer for deep learning models.

    Features:
    - Training loop with validation
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: Optional[str] = None
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10

        logger.info(f"Trainer initialized on {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train model with early stopping.

        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0

                # Save best model
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                self.patience_counter += 1

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if self.patience_counter >= self.max_patience:
                logger.info("Early stopping triggered")
                break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.

    Args:
        data: Input data array
        sequence_length: Length of input sequences
        target_column: Column index for target variable

    Returns:
        X, y arrays
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_column])

    return np.array(X), np.array(y).reshape(-1, 1)
