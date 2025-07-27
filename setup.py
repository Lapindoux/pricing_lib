from setuptools import setup, find_packages

setup(
    name="pricing_lib",
    version="1.0.0",
    author="Basile Simonin",
    author_email="simonin.basile@hotmail.fr",
    description="Librairie de pricing d'options avec Monte Carlo et modèles avancés.",
    long_description=open("README.txt", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ton-repo/pricing_lib",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
