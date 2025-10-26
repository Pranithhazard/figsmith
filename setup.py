"""
FigSmith - Interactive Figure Editor for Scientific Plotting
Setup script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="figsmith",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Interactive figure editor for scientific plotting, specialized for fluid dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/figsmith",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5.0",
        "numpy>=1.20.0",
        "ipywidgets>=8.0.0",
        "IPython>=7.0.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ],
        "io": [
            "h5py>=3.0.0",  # For HDF5 files
            "vtk>=9.0.0",   # For VTK files (optional)
            "pandas>=1.3.0", # For CSV convenience
        ]
    },
    keywords="matplotlib visualization jupyter interactive plotting CFD fluid-dynamics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/figsmith/issues",
        "Source": "https://github.com/yourusername/figsmith",
        "Documentation": "https://figsmith.readthedocs.io",
    },
)
