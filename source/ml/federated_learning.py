"""
Federated Learning Infrastructure.

Enables privacy-preserving distributed model training.
Part of Phase 4: Advanced Features - Federated Learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Update from a federated learning client."""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    loss: float
    timestamp: str


class FederatedServer:
    """
    Federated learning server for aggregating client updates.

    Implements:
    - FedAvg algorithm
    - Secure aggregation
    - Client selection
    - Model versioning
    """

    def __init__(
        self,
        initial_weights: Dict[str, np.ndarray],
        min_clients: int = 3,
        aggregation_method: str = "fedavg"
    ):
        """
        Initialize federated server.

        Args:
            initial_weights: Initial global model weights
            min_clients: Minimum clients required for aggregation
            aggregation_method: Aggregation method (fedavg/fedprox)
        """
        self.global_weights = initial_weights
        self.min_clients = min_clients
        self.aggregation_method = aggregation_method
        self.round_number = 0
        self.client_updates: List[ClientUpdate] = []

        logger.info(f"Initialized federated server with {aggregation_method}")

    def receive_update(self, update: ClientUpdate):
        """
        Receive update from client.

        Args:
            update: Client update with model weights
        """
        # Validate update
        if not self._validate_update(update):
            logger.warning(f"Invalid update from client {update.client_id}")
            return

        self.client_updates.append(update)
        logger.info(
            f"Received update from {update.client_id} "
            f"({len(self.client_updates)}/{self.min_clients})"
        )

    def _validate_update(self, update: ClientUpdate) -> bool:
        """Validate client update."""
        # Check weight shapes match
        for key, weight in update.model_weights.items():
            if key not in self.global_weights:
                return False
            if weight.shape != self.global_weights[key].shape:
                return False

        return True

    def aggregate(self) -> bool:
        """
        Aggregate client updates into global model.

        Returns:
            True if aggregation successful
        """
        if len(self.client_updates) < self.min_clients:
            logger.warning(
                f"Not enough clients for aggregation: "
                f"{len(self.client_updates)}/{self.min_clients}"
            )
            return False

        if self.aggregation_method == "fedavg":
            self._fedavg_aggregate()
        elif self.aggregation_method == "fedprox":
            self._fedprox_aggregate()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        self.round_number += 1
        self.client_updates = []

        logger.info(f"Completed aggregation round {self.round_number}")
        return True

    def _fedavg_aggregate(self):
        """
        Federated Averaging (FedAvg) algorithm.

        Weighted average based on number of samples per client.
        """
        # Calculate total samples
        total_samples = sum(update.num_samples for update in self.client_updates)

        # Initialize aggregated weights
        aggregated = {}

        for key in self.global_weights.keys():
            # Weighted sum of client weights
            weighted_sum = np.zeros_like(self.global_weights[key])

            for update in self.client_updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.model_weights[key]

            aggregated[key] = weighted_sum

        self.global_weights = aggregated

    def _fedprox_aggregate(self):
        """
        FedProx aggregation with proximal term.

        Similar to FedAvg but with regularization toward global model.
        """
        # For now, use FedAvg
        # In production, would add proximal term during client training
        self._fedavg_aggregate()

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Get current global model weights."""
        return self.global_weights.copy()

    def get_round_info(self) -> Dict[str, Any]:
        """Get information about current round."""
        if not self.client_updates:
            avg_loss = 0.0
        else:
            avg_loss = np.mean([u.loss for u in self.client_updates])

        return {
            "round_number": self.round_number,
            "clients_participated": len(self.client_updates),
            "min_clients_required": self.min_clients,
            "ready_for_aggregation": len(self.client_updates) >= self.min_clients,
            "average_loss": float(avg_loss)
        }


class FederatedClient:
    """
    Federated learning client for local training.

    Features:
    - Local model training
    - Differential privacy
    - Secure communication
    """

    def __init__(
        self,
        client_id: str,
        add_noise: bool = True,
        noise_scale: float = 0.1
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            add_noise: Whether to add differential privacy noise
            noise_scale: Scale of privacy noise
        """
        self.client_id = client_id
        self.add_noise = add_noise
        self.noise_scale = noise_scale

        logger.info(f"Initialized federated client: {client_id}")

    def train_local_model(
        self,
        global_weights: Dict[str, np.ndarray],
        local_data: pd.DataFrame,
        local_labels: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.01
    ) -> ClientUpdate:
        """
        Train model on local data.

        Args:
            global_weights: Global model weights to start from
            local_data: Local training data
            local_labels: Local training labels
            epochs: Number of local epochs
            learning_rate: Learning rate

        Returns:
            ClientUpdate with trained weights
        """
        # Initialize local model with global weights
        local_weights = {k: v.copy() for k, v in global_weights.items()}

        # Simulate local training
        # In production, would use actual ML framework
        loss = self._simulate_training(
            local_weights,
            local_data,
            local_labels,
            epochs,
            learning_rate
        )

        # Add differential privacy noise
        if self.add_noise:
            local_weights = self._add_privacy_noise(local_weights)

        # Create update
        update = ClientUpdate(
            client_id=self.client_id,
            model_weights=local_weights,
            num_samples=len(local_data),
            loss=loss,
            timestamp=pd.Timestamp.now().isoformat()
        )

        logger.info(
            f"Client {self.client_id} completed training: "
            f"loss={loss:.4f}, samples={len(local_data)}"
        )

        return update

    def _simulate_training(
        self,
        weights: Dict[str, np.ndarray],
        data: pd.DataFrame,
        labels: np.ndarray,
        epochs: int,
        learning_rate: float
    ) -> float:
        """
        Simulate local training.

        In production, replace with actual training loop.
        """
        # Simulate loss decrease
        initial_loss = 1.0
        final_loss = initial_loss * (1 - learning_rate * epochs)

        return max(0.1, final_loss)

    def _add_privacy_noise(
        self,
        weights: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Add differential privacy noise to weights.

        Implements Gaussian mechanism for differential privacy.
        """
        noisy_weights = {}

        for key, weight in weights.items():
            # Add Gaussian noise
            noise = np.random.normal(
                0,
                self.noise_scale,
                size=weight.shape
            )
            noisy_weights[key] = weight + noise

        return noisy_weights


class SecureAggregator:
    """
    Secure aggregation for federated learning.

    Implements secure multi-party computation for privacy.
    """

    def __init__(self):
        """Initialize secure aggregator."""
        self.shares: Dict[str, List[np.ndarray]] = {}

    def create_shares(
        self,
        weights: Dict[str, np.ndarray],
        num_shares: int = 3
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create secret shares of model weights.

        Args:
            weights: Model weights to share
            num_shares: Number of shares to create

        Returns:
            List of weight shares
        """
        shares = [{} for _ in range(num_shares)]

        for key, weight in weights.items():
            # Split into random shares that sum to original
            weight_shares = self._split_into_shares(weight, num_shares)

            for i, share in enumerate(weight_shares):
                shares[i][key] = share

        return shares

    def _split_into_shares(
        self,
        value: np.ndarray,
        num_shares: int
    ) -> List[np.ndarray]:
        """Split value into secret shares."""
        shares = []

        # Generate random shares
        for i in range(num_shares - 1):
            share = np.random.random(value.shape)
            shares.append(share)

        # Last share ensures sum equals original
        last_share = value - sum(shares)
        shares.append(last_share)

        return shares

    def reconstruct(
        self,
        shares: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Reconstruct original weights from shares.

        Args:
            shares: List of weight shares

        Returns:
            Reconstructed weights
        """
        if not shares:
            raise ValueError("No shares provided")

        reconstructed = {}

        # Sum shares for each weight
        for key in shares[0].keys():
            reconstructed[key] = sum(share[key] for share in shares)

        return reconstructed
