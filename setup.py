# -*- coding: utf-8 -*-
"""
C2C Topology Modeling Framework
Setup script for package installation
"""

from setuptools import setup, find_packages
import os


# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# 读取requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements


setup(
    name="C2C",
    version="1.0.0",
    author="C2C Development Team",
    author_email="xiang.li@sophgo.com",
    description="A topology modeling framework chip-to-chip communication",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="http://10.129.4.209/xiang.li/C2C.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.8",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "streamlit>=1.28.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "c2c-webapp=C2C.scripts.run_webapp:run_webapp",
            "c2c-demo=C2C.examples.basic_demo:main",
            "c2c-validate=C2C.examples.tree_torus_validation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="topology, chip-to-chip, communication, modeling, simulation",
    project_urls={
        "Source": "http://10.129.4.209/xiang.li/C2C.git",
    },
)
