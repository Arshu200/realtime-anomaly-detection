#!/usr/bin/env python3
"""
Setup script for Real-Time Anomaly Detection System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Real-Time Anomaly Detection System for CPU usage monitoring"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="realtime-anomaly-detection",
    version="1.0.0",
    author="Arshu200",
    author_email="arshu200@example.com",
    description="A comprehensive real-time CPU usage anomaly detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arshu200/realtime-anomaly-detection",
    project_urls={
        "Bug Reports": "https://github.com/Arshu200/realtime-anomaly-detection/issues",
        "Source": "https://github.com/Arshu200/realtime-anomaly-detection",
        "Documentation": "https://github.com/Arshu200/realtime-anomaly-detection#readme",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Systems Administration",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anomaly-detector=main:main",
            "anomaly-train=main:main",
            "anomaly-monitor=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "grafana/**/*", "docs/*.md"],
    },
    zip_safe=False,
    keywords=[
        "anomaly detection",
        "cpu monitoring", 
        "time series",
        "prophet",
        "influxdb",
        "grafana",
        "prometheus",
        "monitoring",
        "machine learning",
    ],
)