from setuptools import setup, find_packages


setup(
    name="dengue_prediction",
    version="1.0",
    packages=find_packages(),
    package_dir={
        "dengue-prediction": "dengue_prediction",
    },
)
