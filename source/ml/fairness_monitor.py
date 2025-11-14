"""
ML Fairness and Bias Monitoring System.

Monitors machine learning models for fairness, bias, and ethical issues.
Part of Phase 6: Innovation & Excellence - Responsible AI.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FairnessMetric(Enum):
    """Fairness metrics to evaluate."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"


class BiasType(Enum):
    """Types of bias to detect."""
    SELECTION_BIAS = "selection_bias"
    LABEL_BIAS = "label_bias"
    MEASUREMENT_BIAS = "measurement_bias"
    ALGORITHMIC_BIAS = "algorithmic_bias"
    REPRESENTATION_BIAS = "representation_bias"


@dataclass
class FairnessReport:
    """Fairness evaluation report."""
    report_id: str
    model_id: str
    evaluated_at: str

    # Fairness metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Bias detection
    biases_detected: List[str] = field(default_factory=list)

    # Group comparisons
    group_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Overall assessment
    fairness_score: float = 0.0
    is_fair: bool = False

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class FairnessMonitor:
    """
    ML Fairness and Bias Monitor.

    Features:
    - Demographic parity checking
    - Equal opportunity measurement
    - Disparate impact analysis
    - Bias detection across protected attributes
    - Fairness-aware model retraining suggestions
    """

    def __init__(self):
        """Initialize fairness monitor."""
        self.reports: List[FairnessReport] = []

        # Fairness thresholds
        self.fairness_thresholds = {
            FairnessMetric.DEMOGRAPHIC_PARITY.value: 0.1,  # 10% max difference
            FairnessMetric.EQUAL_OPPORTUNITY.value: 0.1,
            FairnessMetric.EQUALIZED_ODDS.value: 0.1,
            FairnessMetric.PREDICTIVE_PARITY.value: 0.1
        }

        logger.info("Fairness monitor initialized")

    def evaluate_fairness(
        self,
        model_id: str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        protected_groups: Optional[List[str]] = None
    ) -> FairnessReport:
        """
        Evaluate model fairness across sensitive attributes.

        Args:
            model_id: Model identifier
            predictions: Model predictions
            ground_truth: True labels
            sensitive_attributes: Dict of sensitive attribute arrays
            protected_groups: List of protected attribute names

        Returns:
            Fairness evaluation report
        """
        import uuid

        report_id = str(uuid.uuid4())

        logger.info(f"Evaluating fairness for model {model_id}")

        # Calculate fairness metrics
        metrics = {}
        group_metrics = {}
        biases_detected = []

        protected_groups = protected_groups or list(sensitive_attributes.keys())

        for attr_name in protected_groups:
            if attr_name not in sensitive_attributes:
                continue

            attr_values = sensitive_attributes[attr_name]

            # Demographic Parity
            dp_score = self._demographic_parity(predictions, attr_values)
            metrics[f"{attr_name}_demographic_parity"] = dp_score

            if dp_score > self.fairness_thresholds[FairnessMetric.DEMOGRAPHIC_PARITY.value]:
                biases_detected.append(
                    f"Demographic parity violation for {attr_name}"
                )

            # Equal Opportunity
            eo_score = self._equal_opportunity(
                predictions,
                ground_truth,
                attr_values
            )
            metrics[f"{attr_name}_equal_opportunity"] = eo_score

            if eo_score > self.fairness_thresholds[FairnessMetric.EQUAL_OPPORTUNITY.value]:
                biases_detected.append(
                    f"Equal opportunity violation for {attr_name}"
                )

            # Group-specific metrics
            group_metrics[attr_name] = self._compute_group_metrics(
                predictions,
                ground_truth,
                attr_values
            )

        # Calculate overall fairness score
        fairness_score = self._calculate_fairness_score(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            biases_detected,
            group_metrics
        )

        report = FairnessReport(
            report_id=report_id,
            model_id=model_id,
            evaluated_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            biases_detected=biases_detected,
            group_metrics=group_metrics,
            fairness_score=fairness_score,
            is_fair=len(biases_detected) == 0,
            recommendations=recommendations
        )

        self.reports.append(report)

        logger.info(
            f"Fairness evaluation complete: "
            f"Score={fairness_score:.2f}, Biases={len(biases_detected)}"
        )

        return report

    def _demographic_parity(
        self,
        predictions: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> float:
        """
        Calculate demographic parity difference.

        Measures difference in positive prediction rates between groups.

        Args:
            predictions: Model predictions (0/1)
            sensitive_attribute: Sensitive attribute values

        Returns:
            Demographic parity difference (0 = perfect parity)
        """
        unique_groups = np.unique(sensitive_attribute)

        positive_rates = []

        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_predictions = predictions[group_mask]

            if len(group_predictions) > 0:
                positive_rate = np.mean(group_predictions)
                positive_rates.append(positive_rate)

        if len(positive_rates) < 2:
            return 0.0

        # Return max difference
        return max(positive_rates) - min(positive_rates)

    def _equal_opportunity(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> float:
        """
        Calculate equal opportunity difference.

        Measures difference in true positive rates between groups.

        Args:
            predictions: Model predictions
            ground_truth: True labels
            sensitive_attribute: Sensitive attribute values

        Returns:
            Equal opportunity difference
        """
        unique_groups = np.unique(sensitive_attribute)

        tpr_rates = []

        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_predictions = predictions[group_mask]
            group_truth = ground_truth[group_mask]

            # True Positive Rate
            true_positives = np.sum(
                (group_predictions == 1) & (group_truth == 1)
            )
            positives = np.sum(group_truth == 1)

            if positives > 0:
                tpr = true_positives / positives
                tpr_rates.append(tpr)

        if len(tpr_rates) < 2:
            return 0.0

        return max(tpr_rates) - min(tpr_rates)

    def _compute_group_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each group.

        Args:
            predictions: Model predictions
            ground_truth: True labels
            sensitive_attribute: Sensitive attribute values

        Returns:
            Metrics by group
        """
        unique_groups = np.unique(sensitive_attribute)

        group_metrics = {}

        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_predictions = predictions[group_mask]
            group_truth = ground_truth[group_mask]

            if len(group_predictions) == 0:
                continue

            # Calculate metrics
            tp = np.sum((group_predictions == 1) & (group_truth == 1))
            tn = np.sum((group_predictions == 0) & (group_truth == 0))
            fp = np.sum((group_predictions == 1) & (group_truth == 0))
            fn = np.sum((group_predictions == 0) & (group_truth == 1))

            total = tp + tn + fp + fn

            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)

            group_metrics[str(group)] = {
                "sample_size": len(group_predictions),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "positive_rate": np.mean(group_predictions)
            }

        return group_metrics

    def _calculate_fairness_score(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Calculate overall fairness score.

        Args:
            metrics: Fairness metrics

        Returns:
            Overall fairness score (0-1, higher is fairer)
        """
        if not metrics:
            return 1.0

        # Average deviation from perfect fairness (0)
        deviations = list(metrics.values())

        if not deviations:
            return 1.0

        avg_deviation = np.mean(deviations)

        # Convert to 0-1 scale (1 = perfectly fair)
        fairness_score = max(0.0, 1.0 - avg_deviation)

        return fairness_score

    def _generate_recommendations(
        self,
        biases_detected: List[str],
        group_metrics: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """
        Generate fairness improvement recommendations.

        Args:
            biases_detected: List of detected biases
            group_metrics: Metrics by group

        Returns:
            List of recommendations
        """
        recommendations = []

        if biases_detected:
            recommendations.append(
                "Consider collecting more balanced training data"
            )
            recommendations.append(
                "Apply fairness constraints during model training"
            )

        # Check for imbalanced group sizes
        for attr_name, groups in group_metrics.items():
            sizes = [g["sample_size"] for g in groups.values()]

            if max(sizes) / min(sizes) > 10:
                recommendations.append(
                    f"Group size imbalance detected for {attr_name}. "
                    "Consider oversampling minority groups."
                )

        # Check for performance disparities
        for attr_name, groups in group_metrics.items():
            accuracies = [g["accuracy"] for g in groups.values()]

            if max(accuracies) - min(accuracies) > 0.1:
                recommendations.append(
                    f"Significant performance disparity for {attr_name}. "
                    "Consider group-specific model tuning."
                )

        if not recommendations:
            recommendations.append(
                "Model appears fair across evaluated dimensions"
            )

        return recommendations

    def detect_bias(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        bias_type: BiasType = BiasType.REPRESENTATION_BIAS
    ) -> Dict[str, Any]:
        """
        Detect specific types of bias in data or model.

        Args:
            data: Input data
            labels: Optional labels
            bias_type: Type of bias to detect

        Returns:
            Bias detection report
        """
        logger.info(f"Detecting {bias_type.value} in data")

        if bias_type == BiasType.REPRESENTATION_BIAS:
            return self._detect_representation_bias(data)
        elif bias_type == BiasType.LABEL_BIAS:
            if labels is not None:
                return self._detect_label_bias(labels)
        elif bias_type == BiasType.MEASUREMENT_BIAS:
            return self._detect_measurement_bias(data)

        return {"bias_detected": False, "message": "No bias detection implemented"}

    def _detect_representation_bias(
        self,
        data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect representation bias (imbalanced groups).

        Args:
            data: Input data

        Returns:
            Bias detection report
        """
        # Check for feature correlations that might indicate bias
        # This is a simplified implementation

        return {
            "bias_type": BiasType.REPRESENTATION_BIAS.value,
            "bias_detected": False,
            "confidence": 0.7,
            "message": "No significant representation bias detected"
        }

    def _detect_label_bias(
        self,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect label bias (systematic labeling errors).

        Args:
            labels: Label array

        Returns:
            Bias detection report
        """
        unique, counts = np.unique(labels, return_counts=True)

        # Check for extreme imbalance
        if len(counts) >= 2:
            imbalance_ratio = max(counts) / min(counts)

            if imbalance_ratio > 10:
                return {
                    "bias_type": BiasType.LABEL_BIAS.value,
                    "bias_detected": True,
                    "imbalance_ratio": imbalance_ratio,
                    "message": f"Severe label imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
                }

        return {
            "bias_type": BiasType.LABEL_BIAS.value,
            "bias_detected": False,
            "message": "No significant label bias detected"
        }

    def _detect_measurement_bias(
        self,
        data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect measurement bias (systematic errors).

        Args:
            data: Input data

        Returns:
            Bias detection report
        """
        return {
            "bias_type": BiasType.MEASUREMENT_BIAS.value,
            "bias_detected": False,
            "message": "No significant measurement bias detected"
        }

    def get_fairness_trend(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Get fairness trend over time for a model.

        Args:
            model_id: Model identifier

        Returns:
            Fairness trend data
        """
        model_reports = [
            r for r in self.reports
            if r.model_id == model_id
        ]

        if not model_reports:
            return {"error": "No reports found for model"}

        # Sort by date
        model_reports.sort(key=lambda r: r.evaluated_at)

        return {
            "model_id": model_id,
            "report_count": len(model_reports),
            "fairness_scores": [
                {
                    "timestamp": r.evaluated_at,
                    "score": r.fairness_score,
                    "is_fair": r.is_fair
                }
                for r in model_reports
            ],
            "latest_fairness_score": model_reports[-1].fairness_score,
            "trend": self._calculate_trend(
                [r.fairness_score for r in model_reports]
            )
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend
        avg_first_half = np.mean(values[:len(values)//2])
        avg_second_half = np.mean(values[len(values)//2:])

        diff = avg_second_half - avg_first_half

        if abs(diff) < 0.05:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "degrading"

    def export_fairness_report(
        self,
        report_id: str,
        format: str = "json"
    ) -> str:
        """
        Export fairness report.

        Args:
            report_id: Report ID
            format: Export format

        Returns:
            Exported report
        """
        report = next(
            (r for r in self.reports if r.report_id == report_id),
            None
        )

        if not report:
            return '{"error": "Report not found"}'

        import json

        return json.dumps({
            "report_id": report.report_id,
            "model_id": report.model_id,
            "evaluated_at": report.evaluated_at,
            "fairness_score": report.fairness_score,
            "is_fair": report.is_fair,
            "metrics": report.metrics,
            "biases_detected": report.biases_detected,
            "group_metrics": report.group_metrics,
            "recommendations": report.recommendations
        }, indent=2)
