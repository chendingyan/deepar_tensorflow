from setuptools import setup, find_packages

setup(
    name="deepar",
    version="0.0.2",
    description="DeepAR tensorflow 2 implementation",
    author="Jeffrey Gleason",
    author_email="jeffrey.gleason@yonder.co",
    url="https://github.com/NewKnowledge/deepar/tree/master",
    install_requires=[
        "tensorflow-gpu==2.0.0",
        "numpy>=1.15.4,<=1.17.3",
        "pandas>=0.23.4,<=0.25.2",
        "scikit-learn[alldeps]>=0.20.3,<=0.21.3",
        "typing",
    ],
    packages=find_packages(),
)

