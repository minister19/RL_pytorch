import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rl_m19",
    version="0.1.0",
    author="Shuang Gao",
    author_email="gaoshuang.dalian@gmail.com",
    description=('A Reinforcement Learning Framework for training.'),
    license="MIT",
    keywords=['Reinforcement Learning, Cart Pole, Quant, Training Framework'],
    url='https://github.com/minister19/RL_pytorch',
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: MIT License",
    ],
    requires=["setuptools", "wheel"]
)
