"""
Setup configuration for Geo_Sentiment_Climate package.

This package provides enterprise-level air quality data analytics
with advanced ML, API, and deployment capabilities.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename: str):
    """Read requirements from file."""
    requirements_path = this_directory / filename
    with open(requirements_path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("==")
        ]

setup(
    name="geo-sentiment-climate",
    version="2.0.0",
    author="Doğa Aydın",
    author_email="dogaa882@gmail.com",
    description="Enterprise-level air quality data analytics platform with ML and API capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dogaaydinn/Geo_Sentiment_Climate",
    project_urls={
        "Bug Tracker": "https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues",
        "Documentation": "https://github.com/dogaaydinn/Geo_Sentiment_Climate/docs",
        "Source Code": "https://github.com/dogaaydinn/Geo_Sentiment_Climate",
        "LinkedIn": "https://www.linkedin.com/in/dogaaydin/",
    },
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.2",
            "pylint>=3.0.3",
            "bandit>=1.7.6",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.5.3",
        ],
        "ml": [
            "tensorflow>=2.15.0",
            "torch>=2.1.2",
            "transformers>=4.36.2",
            "xgboost>=2.0.3",
            "lightgbm>=4.1.0",
            "catboost>=1.2.2",
        ],
        "cloud": [
            "boto3>=1.34.21",
            "google-cloud-storage>=2.14.0",
            "azure-storage-blob>=12.19.0",
        ],
        "all": [
            "pytest>=7.4.3",
            "black>=23.12.1",
            "sphinx>=7.2.6",
            "tensorflow>=2.15.0",
            "boto3>=1.34.21",
        ],
    },
    entry_points={
        "console_scripts": [
            "geo-climate-ingest=source.data_ingestion:main",
            "geo-climate-preprocess=source.data_preprocessing:main",
            "geo-climate-train=source.ml.model_training:main",
            "geo-climate-serve=source.api.main:start_server",
            "geo-climate-predict=source.ml.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "source": ["config/*.yml", "config/*.yaml", "config/*.json"],
    },
    zip_safe=False,
    keywords=[
        "air-quality",
        "climate",
        "machine-learning",
        "data-science",
        "epa",
        "environmental-data",
        "time-series",
        "geospatial",
        "mlops",
        "fastapi",
        "enterprise",
    ],
    platforms=["any"],
    license="Apache 2.0",
)
