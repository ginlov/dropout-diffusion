from setuptools import setup

setup(
    name="dropout-diffusion",
    py_modules=["dropout_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
