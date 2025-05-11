from setuptools import setup, find_packages

setup(
    name="bvspm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-image",
        "scipy",
    ],
    description="An integration of Matplotlib to handle Bruker/Veeco spm data files.",
    author="Uriel Garcilazo Cruz",
    author_email="garcilazo.uriel@gmail.com",
    url="https://github.com/UGarCil/EyewearTerrains",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust license as needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Specify minimum Python version
)