"""
Setup configuration for Geo Climate Python SDK.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geo-climate-sdk",
    version="1.0.0",
    author="Doğa Aydın",
    author_email="dogaa882@gmail.com",
    description="Official Python SDK for Geo Sentiment Climate API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dogaaydinn/Geo_Sentiment_Climate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    keywords="air-quality, climate, prediction, machine-learning, api-client",
    project_urls={
        "Bug Reports": "https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues",
        "Documentation": "https://docs.geo-climate.com",
        "Source": "https://github.com/dogaaydinn/Geo_Sentiment_Climate",
    },
)
