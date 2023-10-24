import os
from setuptools import setup

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()

setup(
    name="epuc",
    version=0.1,
    license="MIT license",
    description="Analysis of Epistemic Uncertainty Quantification",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Mira Juergens",
    author_email="mira.juergens@ugent.be",
    url="https://github.com/mkjuergens/EpistemicUncertaintyAnalysis",
    packages=["epuc"],
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "setuptools",
        "matplotlib",
        "pandas",
        "seaborn",
        "statsmodels",
        "tqdm",
        "torch"
    ],
    include_package_data=True,
)
