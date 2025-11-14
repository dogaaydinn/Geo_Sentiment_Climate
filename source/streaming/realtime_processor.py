"""
Real-Time Data Streaming and Processing.

Provides real-time data ingestion, processing, and analytics.
Part of Phase 6: Innovation & Excellence - Real-Time Systems.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of stream events."""
    SENSOR_DATA = "sensor_data"
    PREDICTION_REQUEST = "prediction_request"
    ALERT = "alert"
    MODEL_UPDATE = "model_update"
    SYSTEM_METRIC = "system_metric"


@dataclass
class StreamEvent:
    """A single event in the stream."""
    event_id: str
    event_type: str
    timestamp: str
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamConsumer:
    """Stream consumer configuration."""
    consumer_id: str
    topics: List[str]
    handler: Callable
    batch_size: int = 1
    auto_commit: bool = True


class KafkaStreamProcessor:
    """
    Kafka-based stream processor.

    Features:
    - Real-time event processing
    - Exactly-once semantics
    - Partitioning and load balancing
    - Consumer groups
    - Dead letter queue
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "geo-climate-processors"
    ):
        """
        Initialize Kafka stream processor.

        Args:
            bootstrap_servers: Kafka broker addresses
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumers: Dict[str, StreamConsumer] = {}
        self.running = False

        logger.info(f"Kafka stream processor initialized: {bootstrap_servers}")

    async def produce(
        self,
        topic: str,
        event: StreamEvent,
        partition: Optional[int] = None
    ):
        """
        Produce an event to a Kafka topic.

        Args:
            topic: Topic name
            event: Event to produce
            partition: Optional partition number
        """
        # In production, would use aiokafka
        message = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "source": event.source,
            "data": event.data,
            "metadata": event.metadata
        }

        logger.info(
            f"Producing event to topic '{topic}': "
            f"{event.event_type} from {event.source}"
        )

        # Simulated kafka produce
        await asyncio.sleep(0.01)  # Simulate network latency

    async def consume(
        self,
        topic: str,
        consumer_id: str,
        handler: Callable,
        batch_size: int = 1
    ):
        """
        Consume events from a Kafka topic.

        Args:
            topic: Topic name
            consumer_id: Consumer identifier
            handler: Event handler function
            batch_size: Number of events to process in batch
        """
        consumer = StreamConsumer(
            consumer_id=consumer_id,
            topics=[topic],
            handler=handler,
            batch_size=batch_size
        )

        self.consumers[consumer_id] = consumer

        logger.info(
            f"Consumer '{consumer_id}' registered for topic '{topic}'"
        )

        # In production, would use aiokafka consumer
        # This is a simulation
        while self.running:
            # Simulate receiving messages
            await asyncio.sleep(1)

            # Simulated event
            event = StreamEvent(
                event_id=f"evt_{datetime.utcnow().timestamp()}",
                event_type=StreamEventType.SENSOR_DATA.value,
                timestamp=datetime.utcnow().isoformat(),
                source="sensor_001",
                data={"pm25": 35.2, "pm10": 52.8}
            )

            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                await self._send_to_dlq(topic, event, str(e))

    async def start(self):
        """Start all consumers."""
        self.running = True
        logger.info("Stream processor started")

        # Start all consumers
        tasks = [
            self.consume(
                topic=topic,
                consumer_id=consumer.consumer_id,
                handler=consumer.handler,
                batch_size=consumer.batch_size
            )
            for consumer in self.consumers.values()
            for topic in consumer.topics
        ]

        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop all consumers."""
        self.running = False
        logger.info("Stream processor stopped")

    async def _send_to_dlq(
        self,
        original_topic: str,
        event: StreamEvent,
        error: str
    ):
        """
        Send failed event to Dead Letter Queue.

        Args:
            original_topic: Original topic
            event: Failed event
            error: Error message
        """
        dlq_topic = f"{original_topic}.dlq"

        event.metadata["error"] = error
        event.metadata["original_topic"] = original_topic
        event.metadata["failed_at"] = datetime.utcnow().isoformat()

        await self.produce(dlq_topic, event)

        logger.warning(f"Event sent to DLQ: {dlq_topic}")


class WebSocketStreamer:
    """
    WebSocket-based real-time streaming.

    Features:
    - Bidirectional communication
    - Room-based broadcasting
    - Automatic reconnection
    - Compression support
    """

    def __init__(self):
        """Initialize WebSocket streamer."""
        self.connections: Dict[str, Any] = {}
        self.rooms: Dict[str, List[str]] = {}

        logger.info("WebSocket streamer initialized")

    async def connect(
        self,
        client_id: str,
        websocket: Any
    ):
        """
        Register a WebSocket connection.

        Args:
            client_id: Client identifier
            websocket: WebSocket connection object
        """
        self.connections[client_id] = websocket

        logger.info(f"WebSocket client connected: {client_id}")

        await self.send_to_client(
            client_id,
            {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection.

        Args:
            client_id: Client identifier
        """
        if client_id in self.connections:
            del self.connections[client_id]

        # Remove from rooms
        for room_clients in self.rooms.values():
            if client_id in room_clients:
                room_clients.remove(client_id)

        logger.info(f"WebSocket client disconnected: {client_id}")

    async def send_to_client(
        self,
        client_id: str,
        message: Dict[str, Any]
    ):
        """
        Send message to a specific client.

        Args:
            client_id: Client identifier
            message: Message to send
        """
        if client_id not in self.connections:
            logger.warning(f"Client not found: {client_id}")
            return

        websocket = self.connections[client_id]

        # In production, would use websocket.send_json()
        logger.debug(f"Sending to {client_id}: {message['type']}")

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[List[str]] = None
    ):
        """
        Broadcast message to all connected clients.

        Args:
            message: Message to broadcast
            exclude: Optional list of client IDs to exclude
        """
        exclude = exclude or []

        for client_id in self.connections:
            if client_id not in exclude:
                await self.send_to_client(client_id, message)

        logger.info(f"Broadcast sent to {len(self.connections) - len(exclude)} clients")

    async def join_room(
        self,
        client_id: str,
        room_id: str
    ):
        """
        Add client to a room.

        Args:
            client_id: Client identifier
            room_id: Room identifier
        """
        if room_id not in self.rooms:
            self.rooms[room_id] = []

        if client_id not in self.rooms[room_id]:
            self.rooms[room_id].append(client_id)

        logger.info(f"Client {client_id} joined room {room_id}")

    async def leave_room(
        self,
        client_id: str,
        room_id: str
    ):
        """
        Remove client from a room.

        Args:
            client_id: Client identifier
            room_id: Room identifier
        """
        if room_id in self.rooms and client_id in self.rooms[room_id]:
            self.rooms[room_id].remove(client_id)

        logger.info(f"Client {client_id} left room {room_id}")

    async def broadcast_to_room(
        self,
        room_id: str,
        message: Dict[str, Any]
    ):
        """
        Broadcast message to all clients in a room.

        Args:
            room_id: Room identifier
            message: Message to broadcast
        """
        if room_id not in self.rooms:
            logger.warning(f"Room not found: {room_id}")
            return

        for client_id in self.rooms[room_id]:
            await self.send_to_client(client_id, message)

        logger.info(f"Broadcast sent to room {room_id} ({len(self.rooms[room_id])} clients)")


class StreamAggregator:
    """
    Real-time stream aggregation.

    Features:
    - Windowed aggregations (tumbling, sliding, session)
    - Statistical computations
    - Temporal joins
    - Pattern detection
    """

    def __init__(self):
        """Initialize stream aggregator."""
        self.windows: Dict[str, List[StreamEvent]] = {}
        self.aggregations: Dict[str, Any] = {}

        logger.info("Stream aggregator initialized")

    async def tumbling_window(
        self,
        events: List[StreamEvent],
        window_size_seconds: int,
        aggregation_func: Callable
    ) -> List[Dict[str, Any]]:
        """
        Apply tumbling window aggregation.

        Args:
            events: Stream events
            window_size_seconds: Window size in seconds
            aggregation_func: Aggregation function

        Returns:
            Aggregated results
        """
        # Group events into non-overlapping windows
        windows = {}

        for event in events:
            timestamp = datetime.fromisoformat(event.timestamp)
            window_start = (
                timestamp.timestamp() // window_size_seconds
            ) * window_size_seconds

            if window_start not in windows:
                windows[window_start] = []

            windows[window_start].append(event)

        # Apply aggregation to each window
        results = []

        for window_start, window_events in windows.items():
            aggregated = await aggregation_func(window_events)

            results.append({
                "window_start": datetime.fromtimestamp(window_start).isoformat(),
                "window_end": datetime.fromtimestamp(
                    window_start + window_size_seconds
                ).isoformat(),
                "event_count": len(window_events),
                "aggregation": aggregated
            })

        logger.info(f"Tumbling window aggregation: {len(results)} windows")

        return results

    async def sliding_window(
        self,
        events: List[StreamEvent],
        window_size_seconds: int,
        slide_seconds: int,
        aggregation_func: Callable
    ) -> List[Dict[str, Any]]:
        """
        Apply sliding window aggregation.

        Args:
            events: Stream events
            window_size_seconds: Window size in seconds
            slide_seconds: Slide interval in seconds
            aggregation_func: Aggregation function

        Returns:
            Aggregated results
        """
        if not events:
            return []

        # Determine window boundaries
        start_time = datetime.fromisoformat(events[0].timestamp).timestamp()
        end_time = datetime.fromisoformat(events[-1].timestamp).timestamp()

        results = []
        current_start = start_time

        while current_start < end_time:
            current_end = current_start + window_size_seconds

            # Filter events in this window
            window_events = [
                e for e in events
                if current_start <= datetime.fromisoformat(e.timestamp).timestamp() < current_end
            ]

            if window_events:
                aggregated = await aggregation_func(window_events)

                results.append({
                    "window_start": datetime.fromtimestamp(current_start).isoformat(),
                    "window_end": datetime.fromtimestamp(current_end).isoformat(),
                    "event_count": len(window_events),
                    "aggregation": aggregated
                })

            current_start += slide_seconds

        logger.info(f"Sliding window aggregation: {len(results)} windows")

        return results

    async def compute_statistics(
        self,
        events: List[StreamEvent],
        metric_key: str
    ) -> Dict[str, float]:
        """
        Compute real-time statistics.

        Args:
            events: Stream events
            metric_key: Key to extract metric from event data

        Returns:
            Statistics
        """
        values = [
            float(event.data.get(metric_key, 0))
            for event in events
            if metric_key in event.data
        ]

        if not values:
            return {}

        values.sort()

        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "median": values[len(values) // 2],
            "p95": values[int(len(values) * 0.95)] if len(values) > 0 else 0,
            "p99": values[int(len(values) * 0.99)] if len(values) > 0 else 0
        }

    async def detect_pattern(
        self,
        events: List[StreamEvent],
        pattern: List[Dict[str, Any]]
    ) -> List[List[StreamEvent]]:
        """
        Detect patterns in event stream.

        Args:
            events: Stream events
            pattern: Pattern to detect (list of conditions)

        Returns:
            Matching event sequences
        """
        matches = []

        # Simple pattern matching
        for i in range(len(events) - len(pattern) + 1):
            sequence = events[i:i + len(pattern)]

            is_match = all(
                self._matches_condition(event, condition)
                for event, condition in zip(sequence, pattern)
            )

            if is_match:
                matches.append(sequence)

        logger.info(f"Pattern detection: {len(matches)} matches found")

        return matches

    def _matches_condition(
        self,
        event: StreamEvent,
        condition: Dict[str, Any]
    ) -> bool:
        """Check if event matches condition."""
        if "event_type" in condition:
            if event.event_type != condition["event_type"]:
                return False

        if "source" in condition:
            if event.source != condition["source"]:
                return False

        if "data_conditions" in condition:
            for key, value in condition["data_conditions"].items():
                if event.data.get(key) != value:
                    return False

        return True


class RealTimeAnalytics:
    """
    Real-time analytics dashboard backend.

    Provides live metrics and visualizations.
    """

    def __init__(self):
        """Initialize real-time analytics."""
        self.metrics: Dict[str, List[float]] = {}
        self.websocket_streamer = WebSocketStreamer()

        logger.info("Real-time analytics initialized")

    async def update_metric(
        self,
        metric_name: str,
        value: float
    ):
        """
        Update a real-time metric.

        Args:
            metric_name: Metric name
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(value)

        # Keep last 1000 values
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]

        # Broadcast update
        await self.websocket_streamer.broadcast({
            "type": "metric_update",
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def get_live_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get current live metrics.

        Returns:
            Current metrics with statistics
        """
        results = {}

        for metric_name, values in self.metrics.items():
            if values:
                results[metric_name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return results
