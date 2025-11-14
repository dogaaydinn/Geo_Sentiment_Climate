"""
Deep Learning Models for Time-Series Forecasting.

Implements LSTM, GRU, Transformers, and Attention mechanisms.
Part of Phase 4: Advanced Features - Advanced AI.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time-series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for time-series forecasting.

    Architecture:
    - LSTM layers with dropout
    - Attention mechanism (optional)
    - Fully connected output layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(hidden_size)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        """Forward pass."""
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention if enabled
        if self.use_attention:
            lstm_out, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden state
            lstm_out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class GRUModel(nn.Module):
    """
    GRU model for time-series forecasting.

    Lighter alternative to LSTM with similar performance.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize GRU model."""
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        """Forward pass."""
        # GRU
        gru_out, hidden = self.gru(x)

        # Use last hidden state
        out = gru_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence models.

    Allows model to focus on important time steps.
    """

    def __init__(self, hidden_size: int):
        """Initialize attention layer."""
        super(AttentionLayer, self).__init__()

        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        """
        Apply attention mechanism.

        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_size)

        Returns:
            context_vector, attention_weights
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_output)
        attention_scores = torch.softmax(attention_scores, dim=1)

        # Apply attention weights
        context_vector = torch.sum(attention_scores * lstm_output, dim=1)

        return context_vector, attention_scores


class TransformerModel(nn.Module):
    """
    Transformer model for time-series forecasting.

    Uses multi-head attention for capturing complex patterns.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        """Forward pass."""
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Use mean of all positions
        x = torch.mean(x, dim=1)

        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DeepLearningTrainer:
    """
    Trainer for deep learning models.

    Handles training, validation, and prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            device: Device to use (cuda/cpu)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")

        self.model = model.to(device)
        self.device = device

        logger.info(f"Initialized trainer on device: {device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience

        Returns:
            Training history
        """
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    # Load best model
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        return history

    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Evaluate model on validation/test data."""
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                losses.append(loss.item())

        return np.mean(losses)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction.

    Args:
        data: Input data (n_samples, n_features)
        sequence_length: Length of input sequences

    Returns:
        X (sequences), y (targets)
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, -1])  # Predict last feature

    return np.array(X), np.array(y)
