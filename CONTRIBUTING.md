# Contributing to Geo_Sentiment_Climate

Thank you for your interest in contributing to the Geo_Sentiment_Climate project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)
8. [Reporting Bugs](#reporting-bugs)
9. [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to:

- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Geo_Sentiment_Climate.git`
3. Add upstream remote: `git remote add upstream https://github.com/dogaaydinn/Geo_Sentiment_Climate.git`
4. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix bugs and improve code quality
2. **New Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize code performance
6. **Code Review**: Review pull requests

### Contribution Workflow

1. Ensure your fork is up to date:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. Make your changes

4. Run tests:
   ```bash
   pytest tests/ --cov=source
   ```

5. Run linting:
   ```bash
   black source/ tests/
   flake8 source/ tests/
   mypy source/
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```

8. Create a Pull Request

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Example Code Style

```python
from typing import List, Optional

def process_data(
    data: List[float],
    threshold: float = 0.5,
    normalize: bool = True
) -> Optional[List[float]]:
    """
    Process input data with optional normalization.

    Args:
        data: List of numeric values to process
        threshold: Minimum threshold for filtering
        normalize: Whether to normalize the data

    Returns:
        Processed data list or None if input is empty

    Raises:
        ValueError: If threshold is negative
    """
    if not data:
        return None

    if threshold < 0:
        raise ValueError("Threshold must be non-negative")

    # Process data
    result = [x for x in data if x >= threshold]

    if normalize:
        max_val = max(result)
        result = [x / max_val for x in result]

    return result
```

### Commit Message Convention

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add batch prediction endpoint

Implement /predict/batch endpoint for processing multiple
predictions in a single request. Includes request validation
and error handling.

Closes #123
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=source --cov-report=html

# Run specific test file
pytest tests/test_data_ingestion.py

# Run tests matching pattern
pytest -k "test_model"
```

### Writing Tests

```python
import pytest
from source.ml.model_training import ModelTrainer, TrainingConfig

def test_model_trainer_initialization():
    """Test ModelTrainer initializes correctly."""
    config = TrainingConfig(model_type="xgboost")
    trainer = ModelTrainer(config)

    assert trainer.config.model_type == "xgboost"
    assert trainer.model is None

def test_model_training_with_invalid_config():
    """Test ModelTrainer raises error with invalid config."""
    with pytest.raises(ValueError):
        config = TrainingConfig(model_type="invalid_model")
        trainer = ModelTrainer(config)
```

## Pull Request Process

1. **Update Documentation**: Ensure README and docstrings are updated
2. **Add Tests**: Include tests for new features
3. **Update CHANGELOG**: Add entry to CHANGELOG.md
4. **Pass CI Checks**: Ensure all CI/CD checks pass
5. **Request Review**: Request review from maintainers
6. **Address Feedback**: Respond to review comments
7. **Merge**: Maintainers will merge after approval

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests passing
- [ ] No new warnings

## Related Issues
Closes #123
```

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Collect relevant information

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Step 1
2. Step 2
3. See error

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- Package version: [e.g., 2.0.0]

**Additional context**
Any other relevant information
```

## Feature Requests

We welcome feature requests! Please:

1. Check if feature already exists or is planned
2. Explain the use case and benefit
3. Provide examples if possible
4. Discuss design considerations

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of the feature

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other relevant information
```

## Questions?

Feel free to reach out:

- **Email**: dogaa882@gmail.com
- **GitHub Issues**: For bug reports and feature requests
- **LinkedIn**: https://www.linkedin.com/in/dogaaydin/

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

**Thank you for contributing to Geo_Sentiment_Climate! 🎉**
