"""
Model Optimization Module.

Provides model quantization, pruning, and ONNX conversion for faster inference.
Part of Phase 3: Scaling & Optimization - Performance Optimization.
"""

import numpy as np
import onnx
import onnxruntime as ort
from typing import Any, Dict, Optional, Tuple
import pickle
import joblib
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ModelOptimizer:
    """Optimizes ML models for production inference."""

    def __init__(self):
        self.optimized_models = {}

    def quantize_sklearn_model(
        self,
        model: Any,
        quantization_bits: int = 8
    ) -> Any:
        """
        Quantize scikit-learn model for reduced memory footprint.

        Args:
            model: Scikit-learn model
            quantization_bits: Number of bits for quantization (8 or 16)

        Returns:
            Quantized model
        """
        # For tree-based models, we can reduce precision of splits
        if hasattr(model, 'estimators_'):
            # Ensemble model (RandomForest, GradientBoosting, etc.)
            for estimator in model.estimators_:
                self._quantize_tree_estimator(estimator, quantization_bits)
        elif hasattr(model, 'tree_'):
            # Single tree model
            self._quantize_tree_estimator(model, quantization_bits)

        return model

    def _quantize_tree_estimator(self, estimator: Any, bits: int):
        """Quantize individual tree estimator."""
        if hasattr(estimator, 'tree_'):
            tree = estimator.tree_
            if hasattr(tree, 'threshold'):
                # Quantize thresholds
                dtype = np.float16 if bits == 16 else np.float32
                tree.threshold = tree.threshold.astype(dtype)
            if hasattr(tree, 'value'):
                # Quantize values
                dtype = np.float16 if bits == 16 else np.float32
                tree.value = tree.value.astype(dtype)

    def convert_to_onnx(
        self,
        model: Any,
        input_shape: Tuple,
        output_path: str,
        opset_version: int = 13
    ) -> str:
        """
        Convert model to ONNX format for optimized inference.

        Args:
            model: Model to convert
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version

        Returns:
            Path to ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert based on model type
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            return self._convert_pytorch_to_onnx(
                model, input_shape, output_path, opset_version
            )
        elif TF_AVAILABLE and isinstance(model, tf.keras.Model):
            return self._convert_tensorflow_to_onnx(
                model, input_shape, output_path, opset_version
            )
        else:
            # For scikit-learn models, use skl2onnx
            return self._convert_sklearn_to_onnx(
                model, input_shape, output_path, opset_version
            )

    def _convert_pytorch_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple,
        output_path: Path,
        opset_version: int
    ) -> str:
        """Convert PyTorch model to ONNX."""
        model.eval()
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        return str(output_path)

    def _convert_tensorflow_to_onnx(
        self,
        model: Any,
        input_shape: Tuple,
        output_path: Path,
        opset_version: int
    ) -> str:
        """Convert TensorFlow model to ONNX."""
        try:
            import tf2onnx

            # Convert TensorFlow model to ONNX
            spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=opset_version
            )

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(model_proto.SerializeToString())

            return str(output_path)

        except ImportError:
            raise ImportError("tf2onnx is required for TensorFlow to ONNX conversion")

    def _convert_sklearn_to_onnx(
        self,
        model: Any,
        input_shape: Tuple,
        output_path: Path,
        opset_version: int
    ) -> str:
        """Convert scikit-learn model to ONNX."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Define input type
            initial_type = [('float_input', FloatTensorType(input_shape))]

            # Convert to ONNX
            onx = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=opset_version
            )

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())

            return str(output_path)

        except ImportError:
            raise ImportError("skl2onnx is required for sklearn to ONNX conversion")

    def load_onnx_model(self, model_path: str) -> ort.InferenceSession:
        """
        Load ONNX model for inference.

        Args:
            model_path: Path to ONNX model

        Returns:
            ONNX Runtime inference session
        """
        # Create inference session with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        # Load model
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        return session

    def optimize_onnx_model(
        self,
        model_path: str,
        output_path: str,
        optimization_level: str = "all"
    ) -> str:
        """
        Optimize ONNX model with graph optimizations.

        Args:
            model_path: Path to ONNX model
            output_path: Path to save optimized model
            optimization_level: Optimization level (basic/extended/all)

        Returns:
            Path to optimized model
        """
        from onnxruntime.transformers import optimizer

        # Set optimization level
        if optimization_level == "basic":
            opt_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif optimization_level == "extended":
            opt_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            opt_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = opt_level
        sess_options.optimized_model_filepath = output_path

        # Load and optimize
        session = ort.InferenceSession(model_path, sess_options)

        return output_path

    def benchmark_model(
        self,
        model: Any,
        input_data: np.ndarray,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance.

        Args:
            model: Model to benchmark
            input_data: Sample input data
            num_runs: Number of inference runs

        Returns:
            Performance metrics
        """
        import time

        latencies = []

        # Warm-up run
        if isinstance(model, ort.InferenceSession):
            input_name = model.get_inputs()[0].name
            model.run(None, {input_name: input_data})
        else:
            model.predict(input_data)

        # Benchmark runs
        for _ in range(num_runs):
            start_time = time.perf_counter()

            if isinstance(model, ort.InferenceSession):
                input_name = model.get_inputs()[0].name
                model.run(None, {input_name: input_data})
            else:
                model.predict(input_data)

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "std_latency_ms": float(np.std(latencies))
        }
