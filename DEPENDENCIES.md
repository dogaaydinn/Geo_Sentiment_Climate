# Dependency Management & Troubleshooting Guide

## Overview

This document explains the dependency choices, compatibility issues, and troubleshooting steps for the Geo_Sentiment_Climate project.

---

## Critical Fixes Applied

### 1. ❌ `dataprep==0.4.5` → ✅ `ydata-profiling==4.6.4`

**Issue**: `dataprep==0.4.5` does not exist on PyPI.

**Solution**:
- Removed `dataprep` entirely
- Replaced `pandas-profiling` (deprecated) with `ydata-profiling` (the official successor)
- `ydata-profiling` provides the same functionality with better maintenance

**Usage**:
```python
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Air Quality Data Profile")
profile.to_file("report.html")
```

---

### 2. ❌ `auto-sklearn==0.15.0` → ⚠️ Commented Out

**Issue**: `auto-sklearn` only supports Python ≤3.10, not compatible with Python 3.11+.

**Solutions**:

**Option A**: Use Python 3.10
```bash
# Create environment with Python 3.10
conda create -n geo_climate python=3.10
conda activate geo_climate
pip install auto-sklearn==0.15.0
```

**Option B**: Use Alternative AutoML (Recommended for Python 3.11)
```bash
pip install autogluon.tabular>=0.8.0
```

**Usage with AutoGluon**:
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='aqi').fit(train_df)
predictions = predictor.predict(test_df)
```

---

### 3. ❌ `graph-tool==2.58` → ⚠️ Requires Conda

**Issue**: `graph-tool` is not available on PyPI; requires system dependencies.

**Solution**: Use `python-igraph` as alternative (already included).

**To Install graph-tool** (if needed):
```bash
conda install -c conda-forge graph-tool
```

**Alternative with NetworkX** (recommended):
```python
import networkx as nx

# NetworkX is already included and works great for most graph operations
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3)])
```

---

### 4. ❌ `fancyimpute==0.7.0` → ⚠️ Commented Out

**Issue**: `fancyimpute` has compatibility issues with newer Python and scikit-learn versions.

**Solution**: Use scikit-learn's built-in imputers (already implemented in the project).

**Usage**:
```python
from sklearn.impute import IterativeImputer, KNNImputer

# MICE imputation (equivalent to fancyimpute)
imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = imputer.fit_transform(df)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = knn_imputer.fit_transform(df)
```

**Note**: The project already has comprehensive imputation in `source/missing_handle.py`.

---

## CI/CD Workflow Fix

### Slack Notification Issue

**Problem**: Workflow failed with "Need to provide at least one botToken or webhookUrl".

**Solution**: Added conditional check to skip Slack notification if webhook not configured.

**To Enable Slack Notifications**:

1. Create a Slack Incoming Webhook:
   - Go to https://api.slack.com/messaging/webhooks
   - Create webhook for your workspace
   - Copy the webhook URL

2. Add to GitHub Repository Secrets:
   - Go to: Repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `SLACK_WEBHOOK_URL`
   - Value: Your webhook URL (e.g., `https://hooks.slack.com/services/...`)

3. Push a commit to trigger the workflow
   - Slack notifications will now work automatically

**Workflow Behavior**:
- ✅ If `SLACK_WEBHOOK_URL` is set: Sends Slack notification
- ✅ If not set: Logs a message and continues (doesn't fail)

---

## Python Version Compatibility

### Recommended: Python 3.11

The project is configured for Python 3.11, which offers:
- ✅ Best performance (faster than 3.9, 3.10)
- ✅ Modern type hints and features
- ✅ Active support and security updates

### Compatibility Matrix

| Package | Python 3.9 | Python 3.10 | Python 3.11 | Notes |
|---------|------------|-------------|-------------|-------|
| xgboost | ✅ | ✅ | ✅ | Full support |
| lightgbm | ✅ | ✅ | ✅ | Full support |
| catboost | ✅ | ✅ | ✅ | Full support |
| tensorflow | ✅ | ✅ | ✅ | Full support |
| pytorch | ✅ | ✅ | ✅ | Full support |
| auto-sklearn | ✅ | ✅ | ❌ | Use 3.10 or autogluon |
| graph-tool | ⚠️ | ⚠️ | ⚠️ | Conda only |
| fancyimpute | ⚠️ | ⚠️ | ⚠️ | Use sklearn imputers |

---

## Installation Guide

### Recommended Installation (Python 3.11)

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .

# 5. Verify installation
python -c "import source; print('Installation successful!')"
```

### Alternative: Conda Installation (For auto-sklearn)

```bash
# 1. Create conda environment with Python 3.10
conda create -n geo_climate python=3.10
conda activate geo_climate

# 2. Install conda-specific packages
conda install -c conda-forge graph-tool

# 3. Install pip packages
pip install -r requirements.txt

# 4. Install auto-sklearn
pip install auto-sklearn==0.15.0
```

---

## Troubleshooting

### Issue: `pip install` fails with "No matching distribution"

**Check**:
1. Verify Python version: `python --version`
2. Upgrade pip: `pip install --upgrade pip`
3. Check PyPI availability: Visit package on https://pypi.org

**Common Causes**:
- Package version doesn't exist
- Package incompatible with your Python version
- Package deprecated or renamed

### Issue: Import errors after installation

**Solution**:
```bash
# Reinstall with no cache
pip install --no-cache-dir -r requirements.txt

# Or install package in editable mode
pip install -e .
```

### Issue: Conflicting dependencies

**Solution**:
```bash
# Create fresh environment
python -m venv venv_clean
source venv_clean/bin/activate
pip install -r requirements.txt
```

### Issue: TensorFlow or PyTorch installation problems

**For GPU Support**:
```bash
# TensorFlow with GPU
pip install tensorflow[and-cuda]

# PyTorch with GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only** (smaller, faster install):
```bash
pip install tensorflow-cpu
pip install torch torchvision torchaudio
```

---

## Optional Dependencies

### For Development

```bash
pip install pytest pytest-cov black flake8 mypy bandit pre-commit
```

### For Documentation

```bash
pip install sphinx sphinx-rtd-theme mkdocs mkdocs-material
```

### For Notebooks

```bash
pip install jupyter jupyterlab ipywidgets
```

---

## Dependency Updates

### Check for Outdated Packages

```bash
pip list --outdated
```

### Update Specific Package

```bash
pip install --upgrade package-name
```

### Update All Packages (Use with caution)

```bash
pip install --upgrade -r requirements.txt
```

### Best Practice: Test After Updates

```bash
# Run tests after updating
pytest tests/

# Check for breaking changes
python -m source.ml.model_training --help
```

---

## Creating a Minimal Installation

If you only need core ML features without all optional packages:

```bash
# Create minimal requirements
cat > requirements-minimal.txt << EOF
# Core
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2

# ML Models
xgboost==2.0.3
lightgbm==4.1.0
catboost==1.2.2

# API
fastapi==0.108.0
uvicorn[standard]==0.25.0

# Experiment Tracking
mlflow==2.9.2
EOF

pip install -r requirements-minimal.txt
```

---

## Package Alternatives

If a package causes issues, here are drop-in replacements:

| Original | Alternative | Notes |
|----------|-------------|-------|
| pandas-profiling | ydata-profiling | Official successor |
| fancyimpute | sklearn.impute | Built-in, maintained |
| auto-sklearn | autogluon | Better Python 3.11 support |
| graph-tool | networkx | Pure Python, easy install |
| dataprep | ydata-profiling | Better maintained |

---

## Docker Installation (Recommended for Production)

Docker bypasses all dependency issues:

```bash
# Build image
docker build -t geo-climate:latest .

# Run container
docker run -p 8000:8000 geo-climate:latest

# Or use docker-compose
docker-compose up -d
```

---

## Support

If you encounter dependency issues not covered here:

1. **Check GitHub Issues**: https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues
2. **Check Package Documentation**: Visit package's GitHub/PyPI page
3. **Create an Issue**: Provide:
   - Python version (`python --version`)
   - OS and version (`uname -a` or `ver`)
   - Full error message
   - Output of `pip list`

---

## Summary

✅ **All critical dependencies are now compatible with Python 3.11**

✅ **CI/CD pipeline handles missing Slack webhook gracefully**

✅ **Alternative packages provided for all commented-out dependencies**

✅ **Project installs and runs successfully**

The project prioritizes:
- **Stability**: Only maintained, actively-developed packages
- **Compatibility**: Python 3.11 support
- **Flexibility**: Alternatives documented for all edge cases
- **Production-Ready**: Docker for zero-hassle deployment

---

**Last Updated**: 2024
**Python Version**: 3.11.14
**Status**: ✅ All dependency issues resolved
