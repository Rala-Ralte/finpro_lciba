
from setuptools import setup, find_packages

setup(
    name="lciba",
    version="0.1.0",
    description=(
        "LC-IBA: LibraGrad-Enhanced Multimodal Information Bottleneck "
        "Attribution for Vision-Language Models"
    ),
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.6.0",
        "torchvision",
        "numpy<2.0.0",
        "matplotlib>=3.7.0",
        "Pillow",
        "tqdm",
        "pandas",
        "pyyaml",
        "grad-cam",
        "ftfy",
        "regex",
        "transformers",
    ],
)
