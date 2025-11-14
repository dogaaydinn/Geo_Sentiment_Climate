"""
Blockchain-Based Audit Trail System.

Provides immutable, tamper-proof audit logging using blockchain technology.
Part of Phase 6: Innovation & Excellence - Advanced Security.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """A block in the blockchain."""
    index: int
    timestamp: str
    data: Dict[str, Any]
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def calculate_hash(self) -> str:
        """
        Calculate the hash of this block.

        Returns:
            SHA-256 hash of block contents
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)

        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int = 4):
        """
        Mine the block using Proof of Work.

        Args:
            difficulty: Number of leading zeros required
        """
        target = "0" * difficulty

        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()

        logger.info(f"Block mined: {self.hash} (nonce: {self.nonce})")


@dataclass
class Transaction:
    """A transaction to be recorded in the blockchain."""
    transaction_id: str
    timestamp: str
    event_type: str
    actor: str
    resource: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None


class Blockchain:
    """
    Immutable blockchain for audit trail storage.

    Features:
    - Proof of Work consensus
    - SHA-256 cryptographic hashing
    - Chain validation
    - Tamper detection
    - Distributed storage support
    """

    def __init__(self, difficulty: int = 4):
        """
        Initialize blockchain.

        Args:
            difficulty: Mining difficulty (number of leading zeros)
        """
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_transactions: List[Transaction] = []
        self.mining_reward = 1

        # Create genesis block
        self._create_genesis_block()

        logger.info(f"Blockchain initialized with difficulty {difficulty}")

    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis_block = Block(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            data={"message": "Genesis Block - Geo Climate Audit Chain"},
            previous_hash="0"
        )

        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)

        logger.info("Genesis block created")

    def get_latest_block(self) -> Block:
        """
        Get the most recent block in the chain.

        Returns:
            Latest block
        """
        return self.chain[-1]

    def add_transaction(self, transaction: Transaction):
        """
        Add a transaction to pending transactions.

        Args:
            transaction: Transaction to add
        """
        self.pending_transactions.append(transaction)

        logger.debug(
            f"Transaction added: {transaction.event_type} by {transaction.actor}"
        )

    def mine_pending_transactions(self, miner_address: str = "system"):
        """
        Mine all pending transactions into a new block.

        Args:
            miner_address: Address of the miner
        """
        if not self.pending_transactions:
            logger.info("No pending transactions to mine")
            return

        # Create block with pending transactions
        block = Block(
            index=len(self.chain),
            timestamp=datetime.utcnow().isoformat(),
            data={
                "transactions": [
                    {
                        "transaction_id": tx.transaction_id,
                        "timestamp": tx.timestamp,
                        "event_type": tx.event_type,
                        "actor": tx.actor,
                        "resource": tx.resource,
                        "action": tx.action,
                        "details": tx.details
                    }
                    for tx in self.pending_transactions
                ],
                "transaction_count": len(self.pending_transactions)
            },
            previous_hash=self.get_latest_block().hash
        )

        # Mine the block
        block.mine_block(self.difficulty)

        # Add to chain
        self.chain.append(block)

        logger.info(
            f"Block {block.index} mined with {len(self.pending_transactions)} transactions"
        )

        # Clear pending transactions
        self.pending_transactions = []

        # Reward miner (optional - for distributed systems)
        # self.add_transaction(Transaction(..., action="mining_reward"))

    def validate_chain(self) -> bool:
        """
        Validate the entire blockchain.

        Returns:
            True if chain is valid, False if tampered
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Verify current block's hash
            if current_block.hash != current_block.calculate_hash():
                logger.error(
                    f"Block {i} hash mismatch: "
                    f"stored={current_block.hash}, "
                    f"calculated={current_block.calculate_hash()}"
                )
                return False

            # Verify link to previous block
            if current_block.previous_hash != previous_block.hash:
                logger.error(
                    f"Block {i} previous_hash mismatch: "
                    f"stored={current_block.previous_hash}, "
                    f"expected={previous_block.hash}"
                )
                return False

            # Verify proof of work
            if not current_block.hash.startswith("0" * self.difficulty):
                logger.error(f"Block {i} does not meet difficulty requirement")
                return False

        logger.info("Blockchain validation successful")
        return True

    def get_block_by_index(self, index: int) -> Optional[Block]:
        """
        Get block by index.

        Args:
            index: Block index

        Returns:
            Block or None if not found
        """
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None

    def query_transactions(
        self,
        actor: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Transaction]:
        """
        Query transactions from the blockchain.

        Args:
            actor: Filter by actor
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Matching transactions
        """
        results = []

        for block in self.chain[1:]:  # Skip genesis block
            if "transactions" not in block.data:
                continue

            for tx_data in block.data["transactions"]:
                # Apply filters
                if actor and tx_data.get("actor") != actor:
                    continue

                if event_type and tx_data.get("event_type") != event_type:
                    continue

                if start_time:
                    tx_time = datetime.fromisoformat(tx_data["timestamp"])
                    if tx_time < start_time:
                        continue

                if end_time:
                    tx_time = datetime.fromisoformat(tx_data["timestamp"])
                    if tx_time > end_time:
                        continue

                # Reconstruct transaction
                tx = Transaction(
                    transaction_id=tx_data["transaction_id"],
                    timestamp=tx_data["timestamp"],
                    event_type=tx_data["event_type"],
                    actor=tx_data["actor"],
                    resource=tx_data["resource"],
                    action=tx_data["action"],
                    details=tx_data.get("details", {})
                )
                results.append(tx)

        return results

    def export_chain(self, format: str = "json") -> str:
        """
        Export the entire blockchain.

        Args:
            format: Export format (json, summary)

        Returns:
            Exported blockchain data
        """
        if format == "json":
            return json.dumps([
                {
                    "index": block.index,
                    "timestamp": block.timestamp,
                    "data": block.data,
                    "previous_hash": block.previous_hash,
                    "nonce": block.nonce,
                    "hash": block.hash
                }
                for block in self.chain
            ], indent=2)

        elif format == "summary":
            return json.dumps({
                "chain_length": len(self.chain),
                "difficulty": self.difficulty,
                "latest_block": {
                    "index": self.get_latest_block().index,
                    "hash": self.get_latest_block().hash,
                    "timestamp": self.get_latest_block().timestamp
                },
                "valid": self.validate_chain()
            }, indent=2)

        return ""

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get blockchain statistics.

        Returns:
            Statistics dictionary
        """
        total_transactions = sum(
            block.data.get("transaction_count", 0)
            for block in self.chain[1:]
        )

        return {
            "total_blocks": len(self.chain),
            "total_transactions": total_transactions,
            "chain_valid": self.validate_chain(),
            "pending_transactions": len(self.pending_transactions),
            "difficulty": self.difficulty,
            "genesis_timestamp": self.chain[0].timestamp,
            "latest_block": {
                "index": self.get_latest_block().index,
                "hash": self.get_latest_block().hash,
                "timestamp": self.get_latest_block().timestamp
            }
        }


class BlockchainAuditTrail:
    """
    Blockchain-based audit trail for compliance and security.

    Integrates blockchain with audit logging system.
    """

    def __init__(self, blockchain: Optional[Blockchain] = None):
        """
        Initialize blockchain audit trail.

        Args:
            blockchain: Optional existing blockchain instance
        """
        self.blockchain = blockchain or Blockchain(difficulty=4)

        # Auto-mining configuration
        self.auto_mine = True
        self.mining_interval = 10  # Mine every 10 transactions

        logger.info("Blockchain audit trail initialized")

    def log_event(
        self,
        event_type: str,
        actor: str,
        resource: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an event to the blockchain.

        Args:
            event_type: Type of event
            actor: Who performed the action
            resource: What resource was affected
            action: What action was performed
            details: Additional details

        Returns:
            Transaction ID
        """
        import uuid

        transaction_id = str(uuid.uuid4())

        transaction = Transaction(
            transaction_id=transaction_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            details=details or {}
        )

        self.blockchain.add_transaction(transaction)

        # Auto-mine if threshold reached
        if self.auto_mine:
            if len(self.blockchain.pending_transactions) >= self.mining_interval:
                self.blockchain.mine_pending_transactions()

        logger.info(f"Event logged to blockchain: {event_type} by {actor}")

        return transaction_id

    def verify_event(
        self,
        transaction_id: str
    ) -> Dict[str, Any]:
        """
        Verify an event exists in the blockchain.

        Args:
            transaction_id: Transaction ID to verify

        Returns:
            Verification result
        """
        # Search for transaction
        for block in self.blockchain.chain[1:]:
            if "transactions" not in block.data:
                continue

            for tx in block.data["transactions"]:
                if tx["transaction_id"] == transaction_id:
                    return {
                        "verified": True,
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "transaction": tx,
                        "immutable": True
                    }

        # Check pending transactions
        for tx in self.blockchain.pending_transactions:
            if tx.transaction_id == transaction_id:
                return {
                    "verified": True,
                    "pending": True,
                    "immutable": False,
                    "transaction": {
                        "transaction_id": tx.transaction_id,
                        "timestamp": tx.timestamp,
                        "event_type": tx.event_type,
                        "actor": tx.actor
                    }
                }

        return {
            "verified": False,
            "error": "Transaction not found"
        }

    def generate_proof_of_existence(
        self,
        transaction_id: str
    ) -> Optional[str]:
        """
        Generate a proof of existence certificate.

        Args:
            transaction_id: Transaction ID

        Returns:
            Proof certificate (hash)
        """
        verification = self.verify_event(transaction_id)

        if not verification.get("verified"):
            return None

        if verification.get("pending"):
            return None  # Cannot prove existence for pending transactions

        # Create proof
        proof = {
            "transaction_id": transaction_id,
            "block_index": verification["block_index"],
            "block_hash": verification["block_hash"],
            "chain_validated": self.blockchain.validate_chain(),
            "timestamp": datetime.utcnow().isoformat()
        }

        proof_hash = hashlib.sha256(
            json.dumps(proof, sort_keys=True).encode()
        ).hexdigest()

        logger.info(f"Generated proof of existence: {proof_hash}")

        return proof_hash

    def get_actor_history(
        self,
        actor: str
    ) -> List[Dict[str, Any]]:
        """
        Get all events for a specific actor.

        Args:
            actor: Actor ID

        Returns:
            List of actor's transactions
        """
        transactions = self.blockchain.query_transactions(actor=actor)

        return [
            {
                "transaction_id": tx.transaction_id,
                "timestamp": tx.timestamp,
                "event_type": tx.event_type,
                "resource": tx.resource,
                "action": tx.action,
                "details": tx.details
            }
            for tx in transactions
        ]

    def detect_tampering(self) -> Dict[str, Any]:
        """
        Detect any tampering in the blockchain.

        Returns:
            Tampering detection report
        """
        is_valid = self.blockchain.validate_chain()

        if is_valid:
            return {
                "tamper_detected": False,
                "chain_integrity": "intact",
                "message": "No tampering detected. Blockchain is valid."
            }

        # Find tampering location
        for i in range(1, len(self.blockchain.chain)):
            current_block = self.blockchain.chain[i]
            previous_block = self.blockchain.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return {
                    "tamper_detected": True,
                    "tampered_block": i,
                    "tamper_type": "hash_mismatch",
                    "message": f"Block {i} has been tampered with"
                }

            if current_block.previous_hash != previous_block.hash:
                return {
                    "tamper_detected": True,
                    "tampered_block": i,
                    "tamper_type": "chain_break",
                    "message": f"Chain broken at block {i}"
                }

        return {
            "tamper_detected": True,
            "message": "Tampering detected but location unknown"
        }

    def export_audit_report(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> str:
        """
        Export audit report for a date range.

        Args:
            start_date: Start date
            end_date: End date
            format: Report format

        Returns:
            Audit report
        """
        transactions = self.blockchain.query_transactions(
            start_time=start_date,
            end_time=end_date
        )

        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(transactions),
            "chain_integrity": self.blockchain.validate_chain(),
            "events": [
                {
                    "transaction_id": tx.transaction_id,
                    "timestamp": tx.timestamp,
                    "event_type": tx.event_type,
                    "actor": tx.actor,
                    "resource": tx.resource,
                    "action": tx.action
                }
                for tx in transactions
            ]
        }

        return json.dumps(report, indent=2)
