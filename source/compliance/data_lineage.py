"""
Data Lineage Tracking System.

Tracks data flow, transformations, and provenance for compliance and debugging.
Part of Phase 5: Enterprise & Ecosystem - Compliance & Governance.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class DataOperation(Enum):
    """Types of data operations."""
    INGESTION = "ingestion"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    PREDICTION = "prediction"
    EXPORT = "export"
    DELETE = "delete"
    ANONYMIZATION = "anonymization"


class DataSource(Enum):
    """Data source types."""
    API = "api"
    DATABASE = "database"
    FILE_UPLOAD = "file_upload"
    STREAMING = "streaming"
    BATCH = "batch"
    EXTERNAL_API = "external_api"
    SENSOR = "sensor"


@dataclass
class DataEntity:
    """Represents a data entity in the lineage graph."""
    entity_id: str
    entity_type: str  # dataset, model, prediction, user_data
    name: str
    created_at: str
    source: str
    schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class LineageEdge:
    """Represents a transformation/relationship between entities."""
    edge_id: str
    source_entity_id: str
    target_entity_id: str
    operation: str
    operation_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    actor: Optional[str] = None  # User/system that performed operation


@dataclass
class DataLineageRecord:
    """Complete lineage record for a data entity."""
    entity_id: str
    entity_name: str
    entity_type: str
    created_at: str

    # Upstream dependencies
    upstream_entities: List[str] = field(default_factory=list)
    upstream_operations: List[str] = field(default_factory=list)

    # Downstream dependencies
    downstream_entities: List[str] = field(default_factory=list)
    downstream_operations: List[str] = field(default_factory=list)

    # Transformations applied
    transformations: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    data_quality_score: Optional[float] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # Compliance
    contains_pii: bool = False
    retention_policy: Optional[str] = None
    access_restrictions: List[str] = field(default_factory=list)


class DataLineageTracker:
    """
    Tracks data lineage across the system.

    Features:
    - Entity registration and tracking
    - Relationship mapping
    - Transformation history
    - Impact analysis
    - Compliance tracking
    - Visualization support
    """

    def __init__(self, database_manager=None):
        """
        Initialize data lineage tracker.

        Args:
            database_manager: Optional database manager
        """
        self.db = database_manager
        self.entities: Dict[str, DataEntity] = {}
        self.edges: List[LineageEdge] = []

        # Graph structure for efficient traversal
        self.adjacency_list: Dict[str, List[str]] = {}
        self.reverse_adjacency_list: Dict[str, List[str]] = {}

        logger.info("Data lineage tracker initialized")

    def register_entity(
        self,
        entity_type: str,
        name: str,
        source: str,
        schema: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> DataEntity:
        """
        Register a new data entity.

        Args:
            entity_type: Type of entity (dataset, model, prediction, etc.)
            name: Entity name
            source: Data source
            schema: Optional schema definition
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            Created data entity
        """
        entity_id = str(uuid.uuid4())

        entity = DataEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            created_at=datetime.utcnow().isoformat(),
            source=source,
            schema=schema,
            metadata=metadata or {},
            tags=tags or []
        )

        self.entities[entity_id] = entity
        self.adjacency_list[entity_id] = []
        self.reverse_adjacency_list[entity_id] = []

        logger.info(f"Registered entity: {entity_type}/{name} ({entity_id})")

        return entity

    def record_transformation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        operation: DataOperation,
        operation_metadata: Optional[Dict] = None,
        actor: Optional[str] = None
    ) -> LineageEdge:
        """
        Record a data transformation.

        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            operation: Type of operation
            operation_metadata: Operation details
            actor: User/system performing operation

        Returns:
            Created lineage edge
        """
        edge_id = str(uuid.uuid4())

        edge = LineageEdge(
            edge_id=edge_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            operation=operation.value,
            operation_metadata=operation_metadata or {},
            timestamp=datetime.utcnow().isoformat(),
            actor=actor
        )

        self.edges.append(edge)

        # Update adjacency lists
        if source_entity_id in self.adjacency_list:
            self.adjacency_list[source_entity_id].append(target_entity_id)
        if target_entity_id in self.reverse_adjacency_list:
            self.reverse_adjacency_list[target_entity_id].append(source_entity_id)

        logger.debug(
            f"Recorded transformation: {source_entity_id} -> "
            f"{target_entity_id} ({operation.value})"
        )

        return edge

    def get_lineage(
        self,
        entity_id: str,
        depth: int = 5
    ) -> DataLineageRecord:
        """
        Get complete lineage for an entity.

        Args:
            entity_id: Entity ID
            depth: Maximum depth to traverse

        Returns:
            Complete lineage record
        """
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity not found: {entity_id}")

        # Get upstream lineage (sources)
        upstream_entities = self._get_upstream_entities(entity_id, depth)
        upstream_operations = self._get_operations_for_path(
            upstream_entities,
            entity_id
        )

        # Get downstream lineage (consumers)
        downstream_entities = self._get_downstream_entities(entity_id, depth)
        downstream_operations = self._get_operations_for_path(
            [entity_id],
            downstream_entities
        )

        # Get transformations
        transformations = self._get_entity_transformations(entity_id)

        # Build lineage record
        lineage = DataLineageRecord(
            entity_id=entity_id,
            entity_name=entity.name,
            entity_type=entity.entity_type,
            created_at=entity.created_at,
            upstream_entities=upstream_entities,
            upstream_operations=upstream_operations,
            downstream_entities=downstream_entities,
            downstream_operations=downstream_operations,
            transformations=transformations,
            contains_pii=self._check_pii(entity),
            retention_policy=entity.metadata.get("retention_policy")
        )

        return lineage

    def get_upstream_entities(
        self,
        entity_id: str,
        depth: int = 5
    ) -> List[DataEntity]:
        """
        Get all upstream (source) entities.

        Args:
            entity_id: Entity ID
            depth: Maximum depth to traverse

        Returns:
            List of upstream entities
        """
        upstream_ids = self._get_upstream_entities(entity_id, depth)
        return [self.entities[eid] for eid in upstream_ids if eid in self.entities]

    def get_downstream_entities(
        self,
        entity_id: str,
        depth: int = 5
    ) -> List[DataEntity]:
        """
        Get all downstream (consumer) entities.

        Args:
            entity_id: Entity ID
            depth: Maximum depth to traverse

        Returns:
            List of downstream entities
        """
        downstream_ids = self._get_downstream_entities(entity_id, depth)
        return [self.entities[eid] for eid in downstream_ids if eid in self.entities]

    def analyze_impact(
        self,
        entity_id: str,
        operation: str = "delete"
    ) -> Dict[str, Any]:
        """
        Analyze impact of an operation on an entity.

        Args:
            entity_id: Entity ID
            operation: Operation to analyze (delete, modify, etc.)

        Returns:
            Impact analysis results
        """
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity not found: {entity_id}")

        # Get affected entities
        downstream = self._get_downstream_entities(entity_id, depth=10)

        # Categorize by type
        affected_by_type = {}
        for eid in downstream:
            e = self.entities.get(eid)
            if e:
                entity_type = e.entity_type
                if entity_type not in affected_by_type:
                    affected_by_type[entity_type] = []
                affected_by_type[entity_type].append(e.name)

        return {
            "entity_id": entity_id,
            "entity_name": entity.name,
            "operation": operation,
            "impact_summary": {
                "total_affected": len(downstream),
                "affected_by_type": affected_by_type,
                "critical_dependencies": self._find_critical_dependencies(entity_id)
            },
            "recommendations": self._generate_impact_recommendations(
                entity_id,
                downstream
            )
        }

    def trace_to_source(
        self,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """
        Trace data back to original sources.

        Args:
            entity_id: Entity ID to trace

        Returns:
            Trace path to sources
        """
        paths = []
        self._find_source_paths(entity_id, [], paths)

        # Format paths
        formatted_paths = []
        for path in paths:
            formatted_path = []
            for i, eid in enumerate(path):
                entity = self.entities.get(eid)
                if entity:
                    formatted_path.append({
                        "entity_id": eid,
                        "entity_name": entity.name,
                        "entity_type": entity.entity_type,
                        "step": i
                    })
            formatted_paths.append(formatted_path)

        return formatted_paths

    def export_lineage_graph(
        self,
        format: str = "json"
    ) -> str:
        """
        Export lineage graph for visualization.

        Args:
            format: Export format (json, graphml, dot)

        Returns:
            Exported graph data
        """
        if format == "json":
            return self._export_json()
        elif format == "graphml":
            return self._export_graphml()
        elif format == "dot":
            return self._export_dot()
        else:
            return self._export_json()

    def find_entities_with_pii(self) -> List[DataEntity]:
        """
        Find all entities containing PII.

        Returns:
            Entities with PII
        """
        pii_entities = []
        for entity in self.entities.values():
            if self._check_pii(entity):
                pii_entities.append(entity)
        return pii_entities

    def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for data lineage.

        Returns:
            Compliance report
        """
        total_entities = len(self.entities)
        entities_with_pii = len(self.find_entities_with_pii())

        # Count by type
        by_type = {}
        for entity in self.entities.values():
            entity_type = entity.entity_type
            by_type[entity_type] = by_type.get(entity_type, 0) + 1

        # Check retention policies
        entities_with_retention = sum(
            1 for e in self.entities.values()
            if e.metadata.get("retention_policy")
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_entities": total_entities,
            "entities_by_type": by_type,
            "total_transformations": len(self.edges),
            "pii_compliance": {
                "entities_with_pii": entities_with_pii,
                "percentage": (entities_with_pii / total_entities * 100)
                if total_entities > 0 else 0
            },
            "retention_compliance": {
                "entities_with_policy": entities_with_retention,
                "percentage": (entities_with_retention / total_entities * 100)
                if total_entities > 0 else 0
            },
            "recommendations": [
                "Ensure all PII entities have retention policies",
                "Review and document all data transformations",
                "Implement automated lineage tracking in all pipelines"
            ]
        }

    def _get_upstream_entities(
        self,
        entity_id: str,
        depth: int,
        visited: Optional[Set[str]] = None
    ) -> List[str]:
        """Get upstream entities via BFS."""
        if visited is None:
            visited = set()

        if depth == 0 or entity_id in visited:
            return []

        visited.add(entity_id)
        upstream = []

        if entity_id in self.reverse_adjacency_list:
            for parent_id in self.reverse_adjacency_list[entity_id]:
                if parent_id not in visited:
                    upstream.append(parent_id)
                    upstream.extend(
                        self._get_upstream_entities(parent_id, depth - 1, visited)
                    )

        return upstream

    def _get_downstream_entities(
        self,
        entity_id: str,
        depth: int,
        visited: Optional[Set[str]] = None
    ) -> List[str]:
        """Get downstream entities via BFS."""
        if visited is None:
            visited = set()

        if depth == 0 or entity_id in visited:
            return []

        visited.add(entity_id)
        downstream = []

        if entity_id in self.adjacency_list:
            for child_id in self.adjacency_list[entity_id]:
                if child_id not in visited:
                    downstream.append(child_id)
                    downstream.extend(
                        self._get_downstream_entities(child_id, depth - 1, visited)
                    )

        return downstream

    def _get_operations_for_path(
        self,
        path_entities: List[str],
        target: str
    ) -> List[str]:
        """Get operations along a path."""
        operations = []
        for edge in self.edges:
            if (edge.source_entity_id in path_entities or
                edge.target_entity_id == target):
                operations.append(edge.operation)
        return operations

    def _get_entity_transformations(
        self,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """Get all transformations for an entity."""
        transformations = []
        for edge in self.edges:
            if edge.target_entity_id == entity_id:
                transformations.append({
                    "operation": edge.operation,
                    "source_entity": edge.source_entity_id,
                    "timestamp": edge.timestamp,
                    "actor": edge.actor,
                    "metadata": edge.operation_metadata
                })
        return transformations

    def _check_pii(self, entity: DataEntity) -> bool:
        """Check if entity contains PII."""
        pii_indicators = [
            "email", "name", "phone", "address", "ssn",
            "user_id", "ip_address", "personal"
        ]

        # Check entity name and tags
        entity_str = f"{entity.name} {' '.join(entity.tags)}".lower()

        for indicator in pii_indicators:
            if indicator in entity_str:
                return True

        # Check schema
        if entity.schema:
            schema_str = json.dumps(entity.schema).lower()
            for indicator in pii_indicators:
                if indicator in schema_str:
                    return True

        return False

    def _find_critical_dependencies(
        self,
        entity_id: str
    ) -> List[str]:
        """Find critical downstream dependencies."""
        downstream = self.get_downstream_entities(entity_id, depth=3)

        critical = []
        for entity in downstream:
            # Consider production models and user-facing data critical
            if entity.entity_type in ["model", "prediction", "user_data"]:
                if "production" in entity.tags or "critical" in entity.tags:
                    critical.append(entity.name)

        return critical

    def _generate_impact_recommendations(
        self,
        entity_id: str,
        downstream: List[str]
    ) -> List[str]:
        """Generate recommendations for impact mitigation."""
        recommendations = []

        if len(downstream) > 10:
            recommendations.append(
                f"High impact operation - {len(downstream)} entities affected. "
                "Consider staged rollout."
            )

        if len(downstream) > 0:
            recommendations.append(
                "Notify affected downstream system owners before proceeding."
            )

        recommendations.append(
            "Create backup before operation and verify rollback procedure."
        )

        return recommendations

    def _find_source_paths(
        self,
        entity_id: str,
        current_path: List[str],
        all_paths: List[List[str]]
    ):
        """Recursively find all paths to source entities."""
        current_path = current_path + [entity_id]

        # Check if this is a source (no upstream dependencies)
        if entity_id not in self.reverse_adjacency_list or \
           not self.reverse_adjacency_list[entity_id]:
            all_paths.append(current_path)
            return

        # Continue traversing upstream
        for parent_id in self.reverse_adjacency_list[entity_id]:
            if parent_id not in current_path:  # Avoid cycles
                self._find_source_paths(parent_id, current_path, all_paths)

    def _export_json(self) -> str:
        """Export as JSON."""
        graph_data = {
            "entities": [
                {
                    "id": e.entity_id,
                    "name": e.name,
                    "type": e.entity_type,
                    "created_at": e.created_at,
                    "source": e.source,
                    "tags": e.tags
                }
                for e in self.entities.values()
            ],
            "edges": [
                {
                    "id": e.edge_id,
                    "source": e.source_entity_id,
                    "target": e.target_entity_id,
                    "operation": e.operation,
                    "timestamp": e.timestamp
                }
                for e in self.edges
            ]
        }
        return json.dumps(graph_data, indent=2)

    def _export_graphml(self) -> str:
        """Export as GraphML."""
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
        graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        graphml.append('  <graph id="data_lineage" edgedefault="directed">')

        # Nodes
        for entity in self.entities.values():
            graphml.append(f'    <node id="{entity.entity_id}">')
            graphml.append(f'      <data key="name">{entity.name}</data>')
            graphml.append(f'      <data key="type">{entity.entity_type}</data>')
            graphml.append('    </node>')

        # Edges
        for edge in self.edges:
            graphml.append(
                f'    <edge source="{edge.source_entity_id}" '
                f'target="{edge.target_entity_id}">'
            )
            graphml.append(f'      <data key="operation">{edge.operation}</data>')
            graphml.append('    </edge>')

        graphml.append('  </graph>')
        graphml.append('</graphml>')

        return '\n'.join(graphml)

    def _export_dot(self) -> str:
        """Export as DOT (Graphviz) format."""
        dot_lines = ['digraph data_lineage {']
        dot_lines.append('  rankdir=LR;')

        # Nodes
        for entity in self.entities.values():
            label = f"{entity.name}\\n({entity.entity_type})"
            dot_lines.append(
                f'  "{entity.entity_id}" [label="{label}"];'
            )

        # Edges
        for edge in self.edges:
            dot_lines.append(
                f'  "{edge.source_entity_id}" -> "{edge.target_entity_id}" '
                f'[label="{edge.operation}"];'
            )

        dot_lines.append('}')

        return '\n'.join(dot_lines)
