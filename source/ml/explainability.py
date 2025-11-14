"""
Explainable AI Module.

Provides model interpretability and explanation capabilities using SHAP and LIME.
Part of Phase 4: Advanced Features - Explainable AI.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
import pickle

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Comprehensive model explanation using SHAP and LIME.

    Features:
    - Global feature importance with SHAP
    - Local predictions explanations with LIME
    - Trust scores for predictions
    - Automated explanation reports
    - Counterfactual analysis
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        training_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize explainer.

        Args:
            model: Trained model to explain
            feature_names: List of feature names
            training_data: Training data for background distribution
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

        if SHAP_AVAILABLE:
            self._init_shap_explainer()
        else:
            logger.warning("SHAP not available. Install with: pip install shap")

        if LIME_AVAILABLE:
            self._init_lime_explainer()
        else:
            logger.warning("LIME not available. Install with: pip install lime")

    def _init_shap_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            # Determine model type and use appropriate explainer
            model_type = type(self.model).__name__

            if 'XGB' in model_type or 'LightGBM' in model_type or 'CatBoost' in model_type:
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info(f"Initialized TreeExplainer for {model_type}")

            elif hasattr(self.model, 'predict_proba'):
                # General ML models
                if self.training_data is not None:
                    background = shap.sample(self.training_data, 100)
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict,
                        background
                    )
                    logger.info(f"Initialized KernelExplainer for {model_type}")

            else:
                # Linear models
                self.shap_explainer = shap.LinearExplainer(
                    self.model,
                    self.training_data
                )
                logger.info(f"Initialized LinearExplainer for {model_type}")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None

    def _init_lime_explainer(self):
        """Initialize LIME explainer."""
        try:
            if self.training_data is not None:
                self.lime_explainer = lime_tabular.LimeTabularExplainer(
                    training_data=self.training_data.values,
                    feature_names=self.feature_names,
                    mode='regression',
                    verbose=False
                )
                logger.info("Initialized LIME explainer")
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None

    def explain_global(
        self,
        X: pd.DataFrame,
        max_display: int = 20
    ) -> Dict[str, Any]:
        """
        Generate global feature importance explanation.

        Args:
            X: Input features
            max_display: Maximum features to display

        Returns:
            Dictionary with global explanations
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.error("SHAP explainer not available")
            return {"error": "SHAP not available"}

        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X)

            # For tree models, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Calculate mean absolute SHAP values for importance
            importance = np.abs(shap_values).mean(axis=0)

            # Create feature importance ranking
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Get top features
            top_features = feature_importance.head(max_display)

            return {
                "type": "global",
                "method": "shap",
                "feature_importance": top_features.to_dict('records'),
                "top_features": top_features['feature'].tolist(),
                "importance_scores": top_features['importance'].tolist(),
                "total_features": len(self.feature_names)
            }

        except Exception as e:
            logger.error(f"Global explanation failed: {e}")
            return {"error": str(e)}

    def explain_prediction(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        method: str = "shap",
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain individual prediction.

        Args:
            instance: Single instance to explain
            method: Explanation method ('shap' or 'lime')
            num_features: Number of top features to return

        Returns:
            Dictionary with local explanation
        """
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0] if len(instance.shape) > 1 else instance.values
        else:
            instance_array = instance

        if method == "shap":
            return self._explain_with_shap(instance_array, num_features)
        elif method == "lime":
            return self._explain_with_lime(instance_array, num_features)
        else:
            return {"error": f"Unknown method: {method}"}

    def _explain_with_shap(
        self,
        instance: np.ndarray,
        num_features: int
    ) -> Dict[str, Any]:
        """Explain prediction using SHAP."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"error": "SHAP not available"}

        try:
            # Get SHAP values for this instance
            shap_values = self.shap_explainer.shap_values(instance.reshape(1, -1))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Flatten if needed
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # Create explanation dataframe
            explanations = pd.DataFrame({
                'feature': self.feature_names,
                'value': instance,
                'shap_value': shap_values,
                'abs_shap': np.abs(shap_values)
            }).sort_values('abs_shap', ascending=False)

            top_explanations = explanations.head(num_features)

            # Make prediction
            prediction = self.model.predict(instance.reshape(1, -1))[0]

            # Calculate base value (expected value)
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]
            else:
                base_value = 0.0

            return {
                "type": "local",
                "method": "shap",
                "prediction": float(prediction),
                "base_value": float(base_value),
                "explanations": top_explanations[['feature', 'value', 'shap_value']].to_dict('records'),
                "top_positive_features": top_explanations[top_explanations['shap_value'] > 0]['feature'].tolist()[:5],
                "top_negative_features": top_explanations[top_explanations['shap_value'] < 0]['feature'].tolist()[:5]
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": str(e)}

    def _explain_with_lime(
        self,
        instance: np.ndarray,
        num_features: int
    ) -> Dict[str, Any]:
        """Explain prediction using LIME."""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {"error": "LIME not available"}

        try:
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )

            # Get prediction
            prediction = self.model.predict(instance.reshape(1, -1))[0]

            # Extract feature contributions
            lime_values = explanation.as_list()

            explanations = []
            for feature_desc, contribution in lime_values:
                # Parse feature name from description
                feature_name = feature_desc.split('<=')[0].split('>')[0].strip()
                explanations.append({
                    'feature': feature_name,
                    'contribution': contribution,
                    'description': feature_desc
                })

            return {
                "type": "local",
                "method": "lime",
                "prediction": float(prediction),
                "explanations": explanations,
                "intercept": explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0,
                "r2_score": explanation.score if hasattr(explanation, 'score') else None
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"error": str(e)}

    def calculate_trust_score(
        self,
        instance: np.ndarray,
        prediction: float,
        explanation: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate trust score for prediction.

        Based on:
        - Consistency of feature contributions
        - Magnitude of top features
        - Stability of explanation

        Args:
            instance: Input instance
            prediction: Model prediction
            explanation: Explanation dictionary

        Returns:
            Trust metrics
        """
        try:
            if explanation.get("method") != "shap":
                return {"error": "Trust score requires SHAP explanations"}

            explanations_data = explanation.get("explanations", [])
            if not explanations_data:
                return {"trust_score": 0.0, "confidence": "low"}

            # Extract SHAP values
            shap_values = [exp['shap_value'] for exp in explanations_data]

            # Calculate metrics
            total_impact = sum(abs(sv) for sv in shap_values)
            top_3_impact = sum(abs(sv) for sv in shap_values[:3])

            # Concentration score: how much impact is in top features
            concentration = top_3_impact / total_impact if total_impact > 0 else 0

            # Consistency score: ratio of consistent direction
            positive_count = sum(1 for sv in shap_values if sv > 0)
            negative_count = len(shap_values) - positive_count
            consistency = abs(positive_count - negative_count) / len(shap_values)

            # Combined trust score (0-1)
            trust_score = (concentration * 0.6 + consistency * 0.4)

            # Determine confidence level
            if trust_score > 0.7:
                confidence = "high"
            elif trust_score > 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "trust_score": round(trust_score, 3),
                "confidence": confidence,
                "concentration": round(concentration, 3),
                "consistency": round(consistency, 3),
                "total_impact": round(total_impact, 3)
            }

        except Exception as e:
            logger.error(f"Trust score calculation failed: {e}")
            return {"error": str(e)}

    def generate_counterfactual(
        self,
        instance: np.ndarray,
        target_prediction: float,
        max_features_to_change: int = 3
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation.

        Shows what changes would lead to desired prediction.

        Args:
            instance: Original instance
            target_prediction: Desired prediction value
            max_features_to_change: Maximum features to modify

        Returns:
            Counterfactual explanation
        """
        try:
            # Get current prediction and explanation
            current_pred = self.model.predict(instance.reshape(1, -1))[0]
            explanation = self.explain_prediction(instance, method="shap")

            if "error" in explanation:
                return explanation

            # Extract top impactful features
            explanations_data = sorted(
                explanation['explanations'],
                key=lambda x: abs(x['shap_value']),
                reverse=True
            )[:max_features_to_change]

            # Suggest changes
            suggestions = []
            for exp in explanations_data:
                feature_idx = self.feature_names.index(exp['feature'])
                current_value = exp['value']
                shap_value = exp['shap_value']

                # Determine direction of change
                if target_prediction > current_pred:
                    # Need to increase prediction
                    if shap_value > 0:
                        suggested_change = "increase"
                        suggested_value = current_value * 1.2
                    else:
                        suggested_change = "decrease"
                        suggested_value = current_value * 0.8
                else:
                    # Need to decrease prediction
                    if shap_value > 0:
                        suggested_change = "decrease"
                        suggested_value = current_value * 0.8
                    else:
                        suggested_change = "increase"
                        suggested_value = current_value * 1.2

                suggestions.append({
                    'feature': exp['feature'],
                    'current_value': current_value,
                    'suggested_value': suggested_value,
                    'suggested_change': suggested_change,
                    'impact': abs(shap_value)
                })

            return {
                "type": "counterfactual",
                "current_prediction": float(current_pred),
                "target_prediction": float(target_prediction),
                "suggestions": suggestions,
                "num_changes": len(suggestions)
            }

        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return {"error": str(e)}

    def generate_report(
        self,
        instance: np.ndarray,
        output_format: str = "dict"
    ) -> Union[Dict, str]:
        """
        Generate comprehensive explanation report.

        Args:
            instance: Instance to explain
            output_format: 'dict' or 'html'

        Returns:
            Explanation report
        """
        try:
            # Get prediction
            prediction = self.model.predict(instance.reshape(1, -1))[0]

            # Get explanations
            shap_explanation = self.explain_prediction(instance, method="shap")

            # Calculate trust score
            trust = self.calculate_trust_score(instance, prediction, shap_explanation)

            report = {
                "prediction": float(prediction),
                "explanation": shap_explanation,
                "trust_metrics": trust,
                "timestamp": pd.Timestamp.now().isoformat()
            }

            if output_format == "html":
                return self._format_html_report(report)
            else:
                return report

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}

    def _format_html_report(self, report: Dict) -> str:
        """Format report as HTML."""
        html = f"""
        <html>
        <head><title>Model Explanation Report</title></head>
        <body>
            <h1>Model Explanation Report</h1>
            <h2>Prediction: {report['prediction']:.3f}</h2>
            <h3>Trust Score: {report['trust_metrics'].get('trust_score', 'N/A')}
                ({report['trust_metrics'].get('confidence', 'N/A')})</h3>
            <h3>Top Feature Contributions:</h3>
            <table border="1">
                <tr><th>Feature</th><th>Value</th><th>Impact</th></tr>
        """

        for exp in report['explanation'].get('explanations', [])[:10]:
            html += f"""
                <tr>
                    <td>{exp['feature']}</td>
                    <td>{exp['value']:.3f}</td>
                    <td>{exp['shap_value']:.3f}</td>
                </tr>
            """

        html += """
            </table>
            <p>Generated: """ + report['timestamp'] + """</p>
        </body>
        </html>
        """

        return html
